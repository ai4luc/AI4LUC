"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# ********** ResNet50 **********


# Pack
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

# Set up
path_trained_models = '../../trained_models/'
path_datatrain = '../../../data/cerradata/train2/images'

# Data
img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Separando dados para as variaveis de treinamento e validacao
x_train = img.flow_from_directory(path_datatrain,
                                  target_size=(256, 256),
                                  batch_size=128,
                                  color_mode='rgba',
                                  class_mode="categorical",
                                  subset='training')


# Model
resnet50 = tf.keras.applications.ResNet50(include_top=True,
                                          weights=None,

                                          input_shape=(256, 256, 4),
                                          pooling=None,
                                          classes=8)

resnet50.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics='accuracy')


# Controlando a parada do treinamento
stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=8)

# Salvando o melhor modelo
checkpoint_weights = ModelCheckpoint(path_trained_models+"best_weights_resnet50.h5", monitor='loss', verbose=1, save_weights_only=True,
                                     save_best_only=True, mode='auto', period=1)

checkpoint_all = ModelCheckpoint(path_trained_models+"best_model_resnet50.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)


# Treinamento
history = resnet50.fit(x_train, epochs=80, callbacks=[stopping, checkpoint_all])

# Saving last trained_models
resnet50.save_weights(path_trained_models+'resnet50_weights.h5')
configs = resnet50.to_json()
with open(path_trained_models+'resnet50_json.json', 'w') as json_file:
    json_file.write(configs)

resnet50.save_weights(path_trained_models+'resnet50.hdf5')
