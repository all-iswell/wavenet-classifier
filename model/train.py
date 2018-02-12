#! /home/aeatda/anaconda3/envs/proj3/bin/python

"""Training script for modified WaveNet classifier.
Based on binary dataset with crying/laughing labels.

"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, get_tfrecord

BATCH_SIZE = 1
DATA_PATH = '../_data/tfrecords/sample_{}_{}.tfrecord'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 3600
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Number of samples in each tfrecord row.'
                        ' Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many tfrecord rows process at once.'
                        ' Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                        help='The path of the laugh/cry tfrecord'
                        'data.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                        '(execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                        'information for TensorBoard. '
                        'If the model already exists, it will restore '
                        'the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after.'
                        'Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: '
                        + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: '
                        + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: '
                        + WAVENET_PARAMS + '.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer.'
                        'Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. '
                        'Default: False')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be '
                        'kept alive. Default: ' + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Read TFRecords and create network.
    tf.reset_default_graph()

    data_train = get_tfrecord(name='train',
                              sample_size=args.sample_size,
                              batch_size=args.batch_size,
                              seed=None,
                              repeat=None,
                              data_path=args.data_path)
    data_test = get_tfrecord(name='test',
                             sample_size=args.sample_size,
                             batch_size=args.batch_size,
                             seed=None,
                             repeat=None,
                             data_path=args.data_path)

    train_itr = data_train.make_one_shot_iterator()
    test_itr = data_test.make_one_shot_iterator()

    train_batch, train_label = train_itr.get_next()
    test_batch, test_label = test_itr.get_next()

    train_batch = tf.reshape(train_batch, [-1, train_batch.shape[1], 1])
    test_batch = tf.reshape(test_batch, [-1, test_batch.shape[1], 1])

    # Create network.
    net = WaveNetModel(sample_size=args.sample_size,
                       batch_size=args.batch_size,
                       dilations=wavenet_params["dilations"],
                       filter_width=wavenet_params["filter_width"],
                       residual_channels=wavenet_params["residual_channels"],
                       dilation_channels=wavenet_params["dilation_channels"],
                       skip_channels=wavenet_params["skip_channels"],
                       histograms=args.histograms)

    train_loss = net.loss(train_batch, train_label)
    test_loss = net.loss(test_batch, test_label)

    # Optimizer
    # Temporarily set to momentum optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate,
                                           momentum=args.momentum)
    trainable = tf.trainable_variables()
    optim = optimizer.minimize(train_loss, var_list=trainable)

    # Accuracy of test data
    pred_test = net.predict_proba(test_batch, audio_only=True)
    equals = tf.equal(tf.squeeze(test_label), tf.round(pred_test))
    acc = tf.reduce_mean(tf.cast(equals, tf.float32))

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    sess.run([init, init2])

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=args.max_checkpoints)

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()
            if step == saved_global_step + 1:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
            if args.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                summary_, train_loss_, test_loss_, acc_,  _ = sess.run(
                    [summaries, train_loss, test_loss,
                     acc, optim],
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary_, step)
                writer.add_run_metadata(run_metadata,
                                        'step_{:04d}'.format(step))
                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                summary_, train_loss_, test_loss_, acc_,  _ = sess.run(
                    [summaries, train_loss, test_loss,
                     acc, optim],
                    options=run_options,
                    run_metadata=run_metadata)
                writer.add_summary(summary_, step)

            duration = time.time() - start_time
            print("step {:d}:  trainloss = {:.3f}, "
                  "testloss = {:.3f}, acc = {:.3f}, ({:.3f} sec/step)"
                  .format(step, train_loss_, test_loss_, acc_, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step and step > last_saved_step:
            save(saver, sess, logdir, step)
        elif not step:
            print("No training performed during session.")
        else:
            pass


if __name__ == '__main__':
    main()
