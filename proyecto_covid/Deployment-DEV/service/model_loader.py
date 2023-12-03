import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

def loadModelH5():
    MODEL_H5_FILE = "covid_model_full_tf2.h5"
    MODEL_H5_PATH = "/home/dasniel298/models/model/tf2x/keras/full/"

    # Agrega un bloque custom_object_scope para manejar la capa 'KerasLayer'
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
        loaded_model = load_model(MODEL_H5_PATH + MODEL_H5_FILE)

    print(MODEL_H5_FILE, " Loading from disk >> ", loaded_model)

    return loaded_model
