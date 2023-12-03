import tensorflow as tf

# Cargar el modelo
model = tf.keras.models.load_model('/home/dasniel298/models/model/tf2x/keras/full/covid_model_full_tf2.h5"')

# Obtener la representación en cadena del grafo del modelo
print(model.summary())

# Alternativamente, puedes imprimir el grafo TensorFlow directamente
print(model.inputs)
