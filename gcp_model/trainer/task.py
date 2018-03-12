#! /home/aeatda/anaconda3/envs/proj3/bin/python

"""Training script for modified WaveNet classifier.
Based on binary dataset with labels.

"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time

import six
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.saved_model import signature_constants as sig_constants

import trainer.model as model


tf.logging.set_verbosity(tf.logging.INFO)

JOB_DIR_ROOT = './job_dir'
NUM_SAMPLES = 16000
BATCH_SIZE = 40
TRAIN_STEPS = int(1e5)
EVAL_FREQUENCY = 50
# NUM_EPOCHS = 1  # not used
WAVENET_PARAMS = './wavenet_params.json'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
# EPSILON = 0.001
EXPORT_FORMAT = model.EXAMPLE
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MAX_TO_KEEP = 5
# METADATA = False


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--train-files',
                      required=True,
                      type=str,
                      help='Training files local or GCS', nargs='+')
    parser.add_argument('--eval-files',
                      required=True,
                      type=str,
                      help='Evaluation files local or GCS', nargs='+')
    parser.add_argument('--job-dir',
                      type=str,
                      help="""\
                      GCS or local dir for checkpoints, exports, and
                      summaries. Use an existing directory to load a
                      trained model, or a new directory to retrain""")
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES,
                        help='Number of samples in each data row.'
                        ' Default: ' + str(NUM_SAMPLES) + '.')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='How many examples to process at once.'
                        ' Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--train-steps',
                      type=int,
                        default=TRAIN_STEPS,
                      help='Maximum number of training steps to perform.')
    parser.add_argument('--eval-frequency',
                      default=EVAL_FREQUENCY,
                      help='Perform one evaluation per n steps')
    parser.add_argument('--num-epochs',
                      type=int,
                      help='Maximum number of epochs on which to train')
    parser.add_argument('--wavenet-params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: '
                        + WAVENET_PARAMS + '.')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: '
                        + str(LEARNING_RATE) + '.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer.'
                        'Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--export-format',
                      type=str,
                      choices=[model.JSON, model.CSV, model.EXAMPLE],
                      default=EXPORT_FORMAT,
                      help="""\
                      Desired input format for the exported saved_model
                      binary.""")
    parser.add_argument('--verbosity',
                      choices=[
                          'DEBUG',
                          'ERROR',
                          'FATAL',
                          'INFO',
                          'WARN'
                      ],
                      default='INFO',
                      help='Set logging verbosity')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. '
                        'Default: False')
    parser.add_argument('--max-checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be '
                        'kept alive. Default: ' + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def save(saver, sess, job_dir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(job_dir, model_name)
    print('Storing checkpoint to {} ...'.format(job_dir), end="")
    sys.stdout.flush()

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, job_dir):
    print("Trying to restore saved checkpoints from {} ...".format(job_dir),
          end="")

    ckpt = tf.train.get_checkpoint_state(job_dir)
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


def get_default_job_dir(job_dir_root):
    job_dir = os.path.join(job_dir_root, 'train', STARTED_DATESTRING)
    return job_dir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Arrangement
    job_dir_root = JOB_DIR_ROOT

    job_dir = args.job_dir
    if job_dir is None:
        job_dir = get_default_job_dir(job_dir_root)
        print('Using default job_dir: {}'.format(job_dir))

    restore_from = job_dir

    return {
        'job_dir': job_dir,
        'job_dir_root': job_dir_root,
        'restore_from': restore_from
    }



def run(target,
        cluster_spec,
        is_chief,
        train_steps,
        # eval_steps,
        job_dir,
        restore_from,
        train_files,
        eval_files,
        batch_size,  # for both train and eval
        # eval_batch_size,
        num_epochs,
        eval_frequency,
        export_format=model.EXAMPLE,
        num_samples=16000,
        learning_rate=1e-3,
        # epsilon=1e-3, used for Adam optimizer
        momentum=0.9,
        wavenet_params=None,
        filter_width=2,
        sample_rate=16000,
        dilations=[2**x for x in range(10)]*2,
        residual_channels=16,
        dilation_channels=16,
        quantization_channels=256,
        skip_channels=16,
        use_biases=True,
        scalar_input=False,
        initial_filter_width=32,
        histograms=True,
        max_checkpoints=5,
        **kwargs):
    """
    run
    """

    if wavenet_params:
        with open(wavenet_params, 'r') as f:
            wavenet_params = json.load(f)
        for key, val in wavenet_params.items():
            exec("{} = {}".format(key, val))


    net = model.WaveNetModel(num_samples=num_samples,
                             batch_size=batch_size,
                             dilations=dilations,
                             filter_width=filter_width,
                             residual_channels=residual_channels,
                             dilation_channels=dilation_channels,
                             skip_channels=skip_channels,
                             quantization_channels=quantization_channels,
                             use_biases=use_biases,
                             scalar_input=scalar_input,
                             initial_filter_width=initial_filter_width,
                             histograms=histograms)


    train_batch_size, test_batch_size = batch_size, batch_size
    # Features and label tensors as read using filename queue
    _, train_batch, train_labels = model.input_fn(
      train_files,
      num_epochs=num_epochs,
      shuffle=True,
      batch_size=train_batch_size
    )

    _, test_batch, test_labels = model.input_fn(
      eval_files,
      num_epochs=num_epochs,
      shuffle=True,
      batch_size=test_batch_size
    )


    # Returns the training graph and global step tensor
    train_op, global_step_tensor = model.model_fn(
      model.TRAIN,
      net,
      train_batch,
      train_labels,
      learning_rate,
      # epsilon,
      momentum,
    )

    eval_dict = model.model_fn(
      model.EVAL,
      net,
      test_batch,
      test_labels,
    )

    crossent_loss = eval_dict['crossentropy']
    acc_val, acc_update_op = eval_dict['accuracy']
    stream_acc = eval_dict['streaming_accuracy']


    writer = tf.summary.FileWriter(
        os.path.join(job_dir, 'eval_{}'\
                .format(datetime.now().strftime("%y%m%d_%H%M%S"))))
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    init2 = tf.local_variables_initializer()
    sess.run([init, init2])

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(),
                           max_to_keep=max_checkpoints)

    try:
        # Restore variables into session including global_step_tensor
        load(saver, sess, restore_from)
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise


    # Training process
    global_step = global_step_tensor.eval(session=sess)
    # last_saved_step = global_step
    while global_step < train_steps:
        print("global_step is", global_step)
        start_time = time.time()
        do_eval = (global_step % eval_frequency == 0)

        global_step, summ_,  _ = sess.run([global_step_tensor,
                                           summaries,
                                           train_op])

        writer.add_summary(summ_, global_step)

        # Evaluate+log every X steps (at end of step)
        # Also store checkpoint (for now)
        if do_eval:
            tf.logging.info('Starting Evaluation for Step: {}'\
                            .format(global_step))
            crossent, _, stracc = sess.run([crossent_loss,
                                                 acc_update_op,
                                                 stream_acc])
            acc = sess.run([acc_val])
            tf.logging.info("Cross-entropy loss: {:3f}"\
                            .format(crossent))
            tf.logging.info("          Accuracy: {:3f}".format(acc[0]))
            tf.logging.info("Streaming Accuracy: {:3f}"\
                            .format(stracc))
            # Save model checkpoint
            save(saver, sess, job_dir, global_step)

        duration = time.time() - start_time
        print("step {:>3} took {:.3f} sec".format(global_step, duration))

    # After steps are done
    latest_checkpoint = tf.train.latest_checkpoint(job_dir)

    if is_chief:
        build_and_run_exports(latest_checkpoint,
                              job_dir,
                              model.SERVING_INPUT_FUNCTIONS[export_format],
                              net)  # needed to preserve variables
    return



# only export format that works is EXAMPLE
def build_and_run_exports(latest, job_dir, serving_input_fn, wavenetmodel):
  """Given the latest checkpoint file export the saved model.
  Args:
    latest (string): Latest checkpoint file
    job_dir (string): Location of checkpoints and model files
      export path.
    serving_input_fn (function)
    wavenetmodel (WaveNetModel): WaveNetModel where variables are stored
  """

  prediction_graph = tf.Graph()
  exporter = tf.saved_model.builder.SavedModelBuilder(
      os.path.join(job_dir, 'export'))
  with prediction_graph.as_default():
    features, inputs_dict = serving_input_fn()
    prediction_dict = model.model_fn(
        model.PREDICT,
        wavenetmodel,
        features.copy(),
        None,  # labels
        # learning parameters not used in prediction mode 
    )
    saver = tf.train.Saver()

    inputs_info = {
        name: tf.saved_model.utils.build_tensor_info(tensor)
        for name, tensor in six.iteritems(inputs_dict)
    }
    output_info = {
        name: tf.saved_model.utils.build_tensor_info(tensor)
        for name, tensor in six.iteritems(prediction_dict)
    }
    signature_def = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs_info,
        outputs=output_info,
        method_name=sig_constants.PREDICT_METHOD_NAME
    )

  with tf.Session(graph=prediction_graph) as session:
    session.run([tf.local_variables_initializer(),
                 tf.global_variables_initializer()])
    saver.restore(session, latest)
    exporter.add_meta_graph_and_variables(
        session,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            sig_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
        },
        # legacy_init_op=main_op()
    )

  exporter.save()


def dispatch(*args, **kwargs):
  """Parse TF_CONFIG to cluster_spec and call run() method
  TF_CONFIG environment variable is available when running using
  gcloud either locally or on cloud. It has all the information required
  to create a ClusterSpec which is important for running distributed code.
  """

  tf_config = os.environ.get('TF_CONFIG')
  # If TF_CONFIG is not available run local
  if not tf_config:
    return run(target='', cluster_spec=None, is_chief=True, *args, **kwargs)

  tf_config_json = json.loads(tf_config)

  cluster = tf_config_json.get('cluster')
  job_name = tf_config_json.get('task', {}).get('type')
  task_index = tf_config_json.get('task', {}).get('index')

  # If cluster information is empty run local
  if job_name is None or task_index is None:
    return run(target='', cluster_spec=None, is_chief=True, *args, **kwargs)

  cluster_spec = tf.train.ClusterSpec(cluster)
  server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)

  # Wait for incoming connections forever
  # Worker ships the graph to the ps server
  # The ps server manages the parameters of the model.
  #
  # See a detailed video on distributed TensorFlow
  # https://www.youtube.com/watch?v=la_M6bCV91M
  if job_name == 'ps':
    server.join()
    return
  elif job_name in ['master', 'worker']:
    return run(server.target, cluster_spec, is_chief=(job_name == 'master'),
               *args, **kwargs)

# TODO:
# Implement saver, loader, checkpoints, exporting
# ===> dispatch, TF_CONFIG
# LATER: modify for distributive computing

def main():
    args = get_arguments()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    args_dict = vars(args)
    args_dict['job_dir'] = directories['job_dir']
    args_dict['restore_from'] = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    # is_overwritten_training = job_dir != restore_from

    dispatch(**args_dict)


if __name__ == '__main__':
    main()
