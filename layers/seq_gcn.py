from keras import activations, initializers, constraints, regularizers
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

class SequentialGraphCNN(Layer):

    def __init__(self,
                 output_dim,
                 num_filters,
                 seq_len,
                 graph_conv_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(SequentialGraphCNN, self).__init__(**kwargs)

        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.graph_conv_filters = graph_conv_filters
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_initializer.__name__ = kernel_initializer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        if self.num_filters != int(self.graph_conv_filters.shape[0] / self.graph_conv_filters.shape[1]):
            raise ValueError('num_filters does not match with graph_conv_filters dimensions.')

        self.input_dim = input_shape[-1]
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)
    
        self.kernel = []
        self.bias = []
        for step in range(self.seq_len):
            self.kernel += [self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel_{}'.format(step),
                                          trainable = True,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)]
            if self.use_bias:
                self.bias += [self.add_weight(shape=(self.output_dim,),
                                            initializer=self.bias_initializer,
                                            name='bias_{}'.format(step),
                                            trainable = True,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)]
            else:
                self.bias = None
        self.built = True

    def call(self, inputs):
        output = []
        if len(inputs.shape) != 4:
                raise ValueError('x must be 4 dimension tensor'
                                 'Got input shape: ' + str(x.get_shape()))
        for step in range(self.seq_len):
            batch_size = tf.shape(inputs)[0]
            tmp = tf.tensordot(self.graph_conv_filters, inputs[:, step, :, :], axes=[1,1])
            tmp = tf.transpose(tmp, [1,0,2])
            tmp = tf.split(tmp, self.num_filters, axis=1)
            tmp = K.concatenate(tmp, axis=2)
            tmp = K.dot(tmp, self.kernel[step])

            if self.use_bias:
                tmp = K.bias_add(tmp, self.bias[step])
            if self.activation is not None:
                output += [self.activation(tmp)]
        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], input_shape[2], self.output_dim)
        return output_shape

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'num_filters': self.num_filters,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(SequentialGraphCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
