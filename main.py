import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from models import *
from utils import *
import numpy as np

batch_size=32
num_epochs=200

def main():
  model = create_model()

  #load data
  train_x, train_y, context_train, seq_context_train, tf_idf_train = prepare_train_data()
  test_x, test_y, context_test, seq_context_test, tf_idf_test = prepare_test_data()

  #callbacks
  model_checkpoint = ModelCheckpoint(filepath = './model_weight.h5',
                                     monitor='val_loss',
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='min',
                                     save_freq="epoch"
                                     )

  #train a model
  history = model.fit([train_x, context_train, seq_context_train, tf_idf_train], train_y, 
                      validation_data=([test_x, context_test, seq_context_test, tf_idf_test], test_y), 
                      batch_size=batch_size,  
                      epochs=num_epochs, 
                      callbacks=[model_checkpoint]
                      )

  #make a prediction
  model.load_weights('./model_weight.h5')
  prediction = model.predict([test_x, context_test, seq_contest_test, tf_idf_test])

  mae = MAE(answer, prediction)
  rmse = RMSE(answer, prediction)
  wape = WAPE(answer, prediction)

  print("MAE: {:.3f}, RMSE: {:.3f}, WAPE: {:.3f}".format(mae, rmse, wape))

if __name__ == "__main__":
  main()

