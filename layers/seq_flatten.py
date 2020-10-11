from keras.layers import Flatten, Layer
import tensorflow as tf

class Seq_Flatten(Layer):
    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.flatten = Flatten()
        
    def call(self, inputs):
        output = []
        for i in range(self.seq_len):
            output += [self.flatten(inputs[:, i, :, :])]
        output = tf.stack(output, axis=1)
        return output