import numpy as np
import pandas as pd
import seaborn as sns
import keras
from tensorflow.keras import Input, activations
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, GRU, Conv2D, Layer, Embedding, concatenate, Softmax, Multiply, Average
from keras.layers import BatchNormalization, PReLU, cudnn_recurrent
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from layers.seq_gcn import SequentialGraphCNN

seq_len = 8
height = 32
width = 40
n_nodes = height*width
gru_activation = 'elu'
n_features = 1
context_feats = 29
num_filters_dist = 4
num_filters_poi = 4
num_filters_transport = 4
output_dim_first = 64
output_dim_second = 32
gru_units = 512
learning_rate = 0.00015

graph_conv_filters_dist = np.load('filters/distance_matrix.npy')
graph_conv_filters_poi = np.load('filters/poi_matrix.npy')
graph_conv_filters_transport = np.load('filters/transport_matrix.npy')

# Create a model
def create_model():
  
  X_input = Input(shape=(seq_len, n_nodes, n_features))

  context_input = Input(shape=(3,))

  seq_context_input = Input(shape=(seq_len, 3))

  tfidf_input = Input(shape=(height*width, 9))

  context = Embedding(input_dim=context_feats, output_dim=5, input_length=3)(context_input)

  context = Flatten()(context)

  seq_context = Embedding(input_dim=context_feats, output_dim=5, input_length=3)(seq_context_input)

  seq_context = Flatten()(seq_context)

  seq_context = Reshape(target_shape=(seq_len, -1))(seq_context)

  #---------------------------------Distance------------------------------------------
  x_dist = SequentialGraphCNN(output_dim=output_dim_first, num_filters=num_filters_dist, 
                        seq_len=seq_len, graph_conv_filters=graph_conv_filters_dist,
                        activation='relu')(X_input)

  x_dist = SequentialGraphCNN(output_dim=output_dim_second, num_filters=num_filters_dist, 
                        seq_len=seq_len, graph_conv_filters=graph_conv_filters_dist, 
                        activation='relu')(x_dist)

  x_dist = Flatten()(x_dist)

  x_dist = Reshape(target_shape=(seq_len, -1))(x_dist)

  x_dist = concatenate([x_dist, seq_context], axis=2)

  x_dist = cudnn_recurrent.CuDNNGRU(gru_units, activation=gru_activation, return_sequences=False)(x_dist)

  x_dist = Reshape([height*width, 1])(x_dist)

  x_dist = Dense(1, activation=None)(x_dist)

  #---------------------------------POI------------------------------------------
  x_poi = SequentialGraphCNN(output_dim=output_dim_first, num_filters=num_filters_poi, 
                        seq_len=seq_len, graph_conv_filters=graph_conv_filters_poi, 
                        activation='relu')(X_input)

  x_poi = SequentialGraphCNN(output_dim=output_dim_second, num_filters=num_filters_poi, 
                        seq_len=seq_len, graph_conv_filters=graph_conv_filters_poi, 
                        activation='relu')(x_poi)

  x_poi = Flatten()(x_poi)

  x_poi = Reshape(target_shape=(seq_len, -1))(x_poi)

  x_poi = concatenate([x_poi, seq_context], axis=2)

  x_poi = cudnn_recurrent.CuDNNGRU(gru_units, activation=gru_activation, return_sequences=False)(x_poi)

  x_poi = Reshape([height*width, 1])(x_poi)

  x_poi = Dense(1, activation=None)(x_poi)

  #---------------------------------Transportation------------------------------------------
  x_transport = SequentialGraphCNN(output_dim=output_dim_first, num_filters=num_filters_transport, 
                        seq_len=seq_len, graph_conv_filters=graph_conv_filters_transport, 
                        activation='relu')(X_input)

  x_transport = SequentialGraphCNN(output_dim=output_dim_second, num_filters=num_filters_transport, 
                        seq_len=seq_len, graph_conv_filters=graph_conv_filters_transport, 
                        activation='relu')(x_transport)

  x_transport = Flatten()(x_transport)

  x_transport = Reshape(target_shape=(seq_len, -1))(x_transport)

  x_transport = concatenate([x_transport, seq_context], axis=2)

  x_transport = cudnn_recurrent.CuDNNGRU(gru_units, activation=gru_activation, return_sequences=False)(x_transport)

  x_transport = Reshape([height*width, 1])(x_transport)

  x_transport = Dense(1, activation=None)(x_transport)

  #-----------------------------------------

  context = tf.expand_dims(context, 1)
  context = tf.tile(context, [1, 1280, 1])

  x_total = concatenate([x_dist, x_poi, x_transport], axis=2)

  total_context = concatenate([x_total, context, tfidf_input], axis=2)

  total_context = Dense(3, 'relu')(total_context)

  context_atten = Softmax(axis=2)(total_context)

  x_total = Multiply()([x_total, context_atten])

  out = K.sum(x_total, axis=2, keepdims=False)

  model = Model(inputs=[X_input, context_input, seq_context_input, tfidf_input], outputs=out)
  adam = Adam(lr = learning_rate)

  model.compile(loss='mse', optimizer=adam)

  return model