#! /home/aeatda/anaconda3/envs/proj3/bin/python

import tensorflow as tf


data_path0 = '../_data/tfrecords/sample_{}_{}_{}.tfrecord'
sample_size = 16000

def parser(serialized_example):
    features = {
        'idx' : tf.FixedLenFeature([1], tf.int64),
        'samples': tf.FixedLenFeature([sample_size], tf.float32),
        'y' : tf.FixedLenFeature([1], tf.float32)
    }
    parsed_feature = tf.parse_single_example(serialized_example, features)
    samples = parsed_feature['samples']
    y = parsed_feature['y']
    idx = parsed_feature['idx']
    return idx, samples, y


def get_tfrecord(topic='onoff', name='train', sample_size=16000, batch_size=1, buffer_size=5000,
                 repeat=1, seed=None, data_path=data_path0):
    def parser(serialized_example):
        features = {
            'idx' : tf.FixedLenFeature([1], tf.int64),
            'samples': tf.FixedLenFeature([sample_size], tf.float32),
            'y' : tf.FixedLenFeature([1], tf.float32)
        }
        parsed_feature = tf.parse_single_example(serialized_example, features)
        samples = parsed_feature['samples']
        y = parsed_feature['y']
        idx = parsed_feature['idx']
        return idx, samples, y

    dataset = tf.data.TFRecordDataset(data_path.format(topic, sample_size, name)).map(parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size, seed=seed)
    dataset = dataset.repeat(repeat)
    return dataset
