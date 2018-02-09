#! /home/aeatda/anaconda3/envs/proj3/bin/python

import tensorflow as tf

data_path0 = '../_data/tfrecords/sample_{}_{}.tfrecord'


def parser(serialized_example, sample_size):
    features = {
        'samples': tf.FixedLenFeature([sample_size], tf.float32),
        'laugh': tf.FixedLenFeature([1], tf.float32)
    }
    parsed_feature = tf.parse_single_example(serialized_example, features)
    samples = parsed_feature['samples']
    laugh = parsed_feature['laugh']
    return samples, laugh


def get_tfrecord(name, sample_size, batch_size=1, buffer_size=5000,
                 repeat=1, seed=None, data_path=None):
    def parser(serialized_example):
        features = {
            'samples': tf.FixedLenFeature([sample_size], tf.float32),
            'laugh': tf.FixedLenFeature([1], tf.float32)
        }
        parsed_feature = tf.parse_single_example(serialized_example, features)
        samples = parsed_feature['samples']
        laugh = parsed_feature['laugh']
        return samples, laugh

    if data_path:
        data_path0 = data_path
    dataset = tf.data.TFRecordDataset(data_path0.format(sample_size, name))\
                     .map(parser)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size, seed=seed)
    dataset = dataset.repeat(repeat)
    return dataset
