#! /home/aeatda/anaconda3/envs/proj3/bin/python

import tensorflow as tf


data_path0 = '../_data/tfrecords/sample_{}_{}_{}.tfrecord'
num_samples = 16000

def parser(serialized_example):
    features = {
        'idx' : tf.FixedLenFeature([1], tf.int64),
        'samples': tf.FixedLenFeature([num_samples], tf.float32),
        'y' : tf.FixedLenFeature([1], tf.float32)
    }
    parsed_feature = tf.parse_single_example(serialized_example, features)
    samples = parsed_feature['samples']
    y = parsed_feature['y']
    idx = parsed_feature['idx']
    return idx, samples, y


def get_tfrecord(topic='onoff', name='train', num_samples=16000, batch_size=1, buffer_size=5000,
                 repeat=1, seed=None, data_path=data_path0):
    def parser(serialized_example):
        features = {
            'idx' : tf.FixedLenFeature([1], tf.int64),
            'samples': tf.FixedLenFeature([num_samples], tf.float32),
            'y' : tf.FixedLenFeature([1], tf.float32)
        }
        parsed_feature = tf.parse_single_example(serialized_example, features)
        samples = parsed_feature['samples']
        y = parsed_feature['y']
        idx = parsed_feature['idx']
        return idx, samples, y

    dataset = tf.data.TFRecordDataset(data_path.format(topic, num_samples, name)).map(parser)
    dataset = dataset.shuffle(buffer_size, seed=seed)
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batch_size)
    return dataset
