"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# ********** CERRANET V3 **********


# Pack
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

# Data
img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Separando dados para as variaveis de treinamento e validacao
x_train = img.flow_from_directory('/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/train2/images/',
                                  target_size=(256, 256),
                                  batch_size=64,
                                  color_mode='rgb',
                                  class_mode="categorical",
                                  subset='training')

x_valida = img.flow_from_directory('/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/train2/images/',
                                   target_size=(256, 256),
                                   batch_size=64,
                                   color_mode='rgb',
                                   class_mode="categorical",
                                   subset='validation')


# Model
def cerranetv3():
    net = keras.models.Sequential()

    # Camadas convolucionais/ Maxpooling/ Dropout
    net.add(keras.layers.Input(shape=(256, 256, 3)))
    net.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.15))

    # Camada que converte matrizes para vetores
    net.add(keras.layers.Flatten())

    # Camadas Ocultas/Dropout
    net.add(keras.layers.Dense(units=256, activation='relu'))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Dense(units=128, activation='relu'))
    net.add(keras.layers.Dropout(0.10))

    # Camada de Saida
    net.add(keras.layers.Dense(8, activation='softmax'))

    # Compilador: Calcula a taxa de perda; a metrica da validacao; otimizacao da fucao de custo usando SGD
    net.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics='accuracy')

    return net


# Controlando a parada do treinamento
stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=8)

# Salvando o melhor modelo
path_trained_models = '/scratch/ideeps/mateus.miranda/ai4luc/trained_models/'
checkpoint_weights = ModelCheckpoint(path_trained_models+"best_weights_cerranetv3.h5", monitor='loss', verbose=1, save_weights_only=True,
                                     save_best_only=True, mode='auto', period=1)

checkpoint_all = ModelCheckpoint(path_trained_models+"best_model_cerranetv3.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)


cerranet = cerranetv3()

print(cerranet.summary())

# Treinamento
path_trained_models = '/scratch/ideeps/mateus.miranda/ai4luc/trained_models/'
history = cerranet.fit(x_train, epochs=80, callbacks=[stopping, checkpoint_all],
                       validation_data=x_valida)

# Saving last trained_models
cerranet.save_weights(path_trained_models+'cerranet_weights.h5')
configs = cerranet.to_json()
with open(path_trained_models+'cerranet_json.json', 'w') as json_file:
    json_file.write(configs)

cerranet.save_weights(path_trained_models+'cerranetv3.hdf5')
