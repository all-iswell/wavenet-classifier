#! /home/aeatda/anaconda3/envs/proj3/bin/python

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import re
import pydub as pdb


DATA_PATH0 = '../_data/tfrecords/{}_16000_{}.tfrecord'
NUM_SAMPLES = 16000


def get_array(filetype='wav'):
    namelist, dflist, ylist = [], [], []
    for category in ['on', 'off']:
        cat_path = '../_data/{}/'.format(category)
        names = [name for name in os.listdir(cat_path)
                 if re.search(r'\.{}$'.format(filetype), name)]
        for name in names:
            f1 = pdb.AudioSegment.from_wav(cat_path+name)\
                    .get_array_of_samples().tolist()
            if len(f1) == 16000:
                namelist.append(name[:-4])
                dflist.append(f1)
                ylist.append(category == 'on')
    return namelist, dflist, ylist


def mu_encode(x, mu=255, bitdepth=16):
    x = np.divide(x, np.power(2, bitdepth-1))
    x_scale = np.sign(x) * np.log(1 + mu*np.abs(x)) / np.log(1 + mu)
    return np.round((x_scale + 1) / 2 * mu)


def write_tfrecord(arr, topic='onoff', name='train'):
    data_path = DATA_PATH0.format(topic, name)
    print(data_path)

    with tf.python_io.TFRecordWriter(data_path) as writer:
        for row in arr:
            idx, samples, y = row[0], row[1:-1], row[-1]

            example = tf.train.Example()
            example.features.feature["idx"].int64_list.value.append(idx)
            example.features.feature["samples"]\
                   .float_list.value.extend(samples)
            example.features.feature["y"].float_list.value.append(y)

            writer.write(example.SerializeToString())
    return


def parser(serialized_example):
    features = {
        'idx': tf.FixedLenFeature([1], tf.int64),
        'samples': tf.FixedLenFeature([NUM_SAMPLES], tf.float32),
        'y': tf.FixedLenFeature([1], tf.float32)
    }
    parsed_feature = tf.parse_single_example(serialized_example, features)
    samples = parsed_feature['samples']
    y = parsed_feature['y']
    idx = parsed_feature['idx']
    return idx, samples, y


def get_tfrecord(topic='onoff', name='train', batch_size=1, buffer_size=5000,
                 repeat=1, seed=None):
    dataset = tf.data.TFRecordDataset(DATA_PATH0.format(topic, name))\
            .map(parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size, seed=seed)
    dataset = dataset.repeat(repeat)
    return dataset


def main(*args):
    namelist, dflist, ylist = get_array()

    df_mu = pd.DataFrame(mu_encode(np.array(dflist)), columns=range(1, 16001))
    df_mu.insert(0, 0, namelist)
    df_mu.insert(16001, 16001, ylist)

    df_s = df_mu.sample(frac=1).reset_index()
    del df_s[0]

    # 2480 / 880 / 880 split
    df_train = df_s.iloc[:2480, :].copy()
    df_eval = df_s.iloc[2480:3360, :].copy()
    df_test = df_s.iloc[3360:4240, :].copy()

    df_train_val = df_train.values.astype(np.int64)
    df_eval_val = df_eval.values.astype(np.int64)
    df_test_val = df_test.values.astype(np.int64)

    write_tfrecord(df_train_val, name='train')
    write_tfrecord(df_eval_val, name='eval')
    write_tfrecord(df_test_val, name='test')

    return

# Code to test getting TFRecord
# data_train = get_tfrecord(batch_size=1, seed=0)
# itr = data_train.make_one_shot_iterator()
# idx, sample_batch, y = itr.get_next()
# sample_batch = tf.reshape(sample_batch, [-1, sample_batch.shape[1], 1])
#
# sess = tf.Session()
# idces, batches, ys = [], [], []
#
# i, bat, yy = sess.run([idx, sample_batch, y])
# idces.extend(i)
# batches.extend(bat)
# ys.extend(yy)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)
    print("Done")
