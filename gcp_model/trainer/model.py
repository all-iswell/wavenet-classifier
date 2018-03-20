#! /home/aeatda/anaconda3/envs/proj3/bin/python

import tensorflow as tf
import json

from .ops import causal_conv


TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'

CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]

FEATURES = {
    'idx' : tf.FixedLenFeature([1], tf.int64),
    'samples': tf.FixedLenFeature([16000], tf.float32),
    'y' : tf.FixedLenFeature([1], tf.float32)
}


# Helper functions for WaveNetModel class
def create_variable(name, shape):
    '''Create a convolution filter variable with the specified name and shape,
    and initialize it using Xavier initialization.'''
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    '''Create a bias variable with the specified name and shape and initialize
    it to zero.'''
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


# scalar_input option does not work for now
class WaveNetModel(object):
    '''WaveNet model modified for binary classification.
    Modified by John Choi (isnbh0)

    Default parameters:
        dilations = [2**i for i in range(10)] * 2
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 16  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

    def __init__(self,
                 num_samples,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2**8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32,
                 histograms=True):
        '''Initializes the WaveNet model.

        Args:
            num_samples: number of samples in each audio file.
            batch_size: audio files per batch (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: # filters to learn for the residual.
            dilation_channels: # filters to learn for the dilated convolution.
            skip_channels: # filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: # amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            scalar_input: Whether to use the quantized waveform directly as
            |   input to the network instead of one-hot encoding it.
            |   Default: False.
            *-initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: True.


        '''
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms

        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        var = dict()

        with tf.variable_scope('wavenet'):
            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_channels = self.quantization_channels
                    initial_filter_width = self.filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_variable(
                    'postprocess1',
                    [1, self.skip_channels, self.skip_channels])
                current['postprocess2'] = create_variable(
                    'postprocess2',
                    [1, self.skip_channels, 1])  ## returns scalar value
                if self.use_biases:
                    current['postprocess1_bias'] = create_bias_variable(
                        'postprocess1_bias',
                        [self.skip_channels])
                    current['postprocess2_bias'] = create_bias_variable(
                        'postprocess2_bias',
                        [1])
                var['postprocessing'] = current

        return var

    def _create_causal_layer(self, input_batch):
        '''Creates a single causal convolution layer.

        The layer can change the number of channels.
        '''
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               output_width):
        '''Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        '''
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.summary.histogram(layer + '_filter', weights_filter)
            tf.summary.histogram(layer + '_gate', weights_gate)
            tf.summary.histogram(layer + '_dense', weights_dense)
            tf.summary.histogram(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.summary.histogram(layer + '_biases_filter', filter_bias)
                tf.summary.histogram(layer + '_biases_gate', gate_bias)
                tf.summary.histogram(layer + '_biases_dense', dense_bias)
                tf.summary.histogram(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        '''Perform convolution for a single convolutional processing step.'''
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def _create_network(self, input_batch):
        '''Construct the WaveNet network.'''
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        current_layer = self._create_causal_layer(current_layer)

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.summary.histogram('postprocess1_weights', w1)
                tf.summary.histogram('postprocess2_weights', w2)
                if self.use_biases:
                    tf.summary.histogram('postprocess1_biases', b1)
                    tf.summary.histogram('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

            ## ADDED CODE HERE. ******************************
            ## RELU activation before going to dense node
            transformed3 = tf.nn.relu(conv2)

            ## Reshape to 2-D tensor with dimensions
            ## batch_size, num_samples
            ## shape[1] constant here so we can use dense
            shape = [tf.shape(transformed3)[0],
                     self.num_samples - sum(self.dilations) - 1]
            transformed3 = tf.reshape(transformed3, shape)

            ## Add dense layer to transform from
            ## batch_size x num_samples 2-D array to
            ## batch_size 1-D array
            final = tf.layers.dense(transformed3, units=1, reuse=tf.AUTO_REUSE,
                                    name='final_out')
        return final

    def _one_hot(self, input_batch, batch_size=None):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        if not batch_size:
            batch_size = self.batch_size

        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [-1,
                     self.num_samples,
                     self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def predict_proba(self, input_batch, name='wavenet'):
        '''Computes the probability of waveform being LAUGHTER.
           Necessary for evaluation with test data.'''

        logits = self.loss_or_logits(input_batch,
                                    get_loss=False,
                                    get_logits=True)['logits']
        logits = tf.reshape(logits, [-1])
        out = tf.nn.sigmoid(logits)
        return out

    def loss_or_logits(self,
             input_batch,
             input_label=None, get_loss=True, get_logits=False,
             name='wavenet'):
        '''Creates a WaveNet network and
        returns crossentropy loss and/or logits.
        If no input_label is given, return logits only.
        The variables are all scoped to the given name.
        '''
        if input_label is None:
            get_loss, get_logits = False, True

        outputs = {}

        with tf.name_scope(name):
            input_audio = tf.to_int32(input_batch)
            encoded = self._one_hot(input_audio)
            if get_loss:
                input_label = tf.reshape(input_label, [-1, 1, 1])

            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded

            ## shape of raw_output is (batch_size, 1)
            raw_output = self._create_network(network_input)

            if get_logits:
                outputs['logits'] = tf.squeeze(raw_output)
            if get_loss:
                with tf.name_scope('loss'):
                    ## format the target values
                    target_output = tf.squeeze(input_label)
                    ## logits
                    prediction = tf.squeeze(raw_output)

                    loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=prediction,
                        labels=target_output)
                    reduced_loss = tf.reduce_mean(loss)

                    # tf.summary.scalar('loss', reduced_loss)

                    outputs['loss'] = reduced_loss
        return outputs


def parser(serialized_example):
    parsed_feature = tf.parse_single_example(serialized_example, FEATURES)
    samples = parsed_feature['samples']
    y = parsed_feature['y']
    idx = parsed_feature['idx']
    return idx, samples, y


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             batch_size=40):
    """
    Generate features and labels for training or evaluation.
    Args:
        filenames: str list of TFRecord files to read from
        num_epochs: how many times to loop through the data
        shuffle: whether to shuffle data
        batch_size: size of batches used in gradient calculation
    Returns:
        (audio, labels) tuple
    """
    dataset = tf.data.TFRecordDataset(filenames).map(parser)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    itr = dataset.make_one_shot_iterator()
    idx, samples, label = itr.get_next()
    return idx, samples, label


def model_fn(mode,
             net,  # WaveNetModel
             input_batch,
             labels,
             learning_rate=1e-3,
             epsilon=1e-4,
             momentum=0.9):
    """ Model fn"""

    outs = net.loss_or_logits(input_batch, labels,
                              get_loss=True, get_logits=True)

    if mode in (PREDICT, EVAL):
        probs = tf.nn.sigmoid(outs['logits'])
        pred = tf.round(probs)

    if mode in (TRAIN, EVAL):
        global_step = tf.train.get_or_create_global_step()

    if mode == PREDICT:
        return {
            'predictions': pred,
            'confidence_1': probs
        }

    if mode == TRAIN:
        tf.summary.scalar('train_loss', outs['loss'])
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           epsilon=epsilon)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
        #                                        momentum=momentum)
        trainable = tf.trainable_variables()
        train_op = optimizer.minimize(outs['loss'],
                                      var_list=trainable,
                                      global_step=global_step)
        return train_op, global_step

    if mode == EVAL:
        tf.summary.scalar('eval_loss', outs['loss'])
        # Return accuracy and cross entropy
        accuracy = tf.metrics.accuracy(labels, pred)

        equals = tf.equal(tf.squeeze(labels), pred)
        streaming_accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))

        return {
            'crossentropy': outs['loss'],
            'accuracy': accuracy,
            'streaming_accuracy': streaming_accuracy
        }

# def csv_serving_input_fn(default_batch_size=None):
def example_serving_input_fn(default_batch_size=None):
    example_bytestring = tf.placeholder(
        shape=[default_batch_size],
        dtype=tf.string,
    )
    features = tf.parse_example(example_bytestring, FEATURES)

    return features['samples'], {'example': example_bytestring}

# def json_serving_input_fn(default_batch_size=None):
SERVING_INPUT_FUNCTIONS = {
# JSON: json_serving_input_fn,
# CSV: csv_serving_input_fn,
EXAMPLE: example_serving_input_fn
}
