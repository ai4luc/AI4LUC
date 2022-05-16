# Cerranet v3: Multiclass cerrado

# Pack
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint


# Data
img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Separando dados para as variaveis de treinamento e validacao
x_train = img.flow_from_directory('/Users/mateus.miranda/INPE-CAP/mestrado/LUCIai project/algorithm/cerraNet_v3/data/dataset_cerradov3_NIR+G+B_splited_50k/train',
                                  target_size=(256, 256),
                                  batch_size=128,
                                  class_mode="categorical")

x_valida = img.flow_from_directory('/Users/mateus.miranda/INPE-CAP/mestrado/LUCIai project/algorithm/cerraNet_v3/data/dataset_cerradov3_NIR+G+B_splited_50k/val',
                                   target_size=(256, 256),
                                   batch_size=128,
                                   class_mode="categorical")


# Model
def cerranetv3():
    net = keras.models.Sequential()

    # Camadas convolucionais/ Maxpooling/ Dropout
    net.add(keras.layers.Input(shape=(256, 256, 3)))
    net.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.10))

    net.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.10))

    net.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.10))

    net.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.10))

    net.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.10))

    net.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    net.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    net.add(keras.layers.Dropout(0.150))

    # Camada que converte matrizes para vetores
    net.add(keras.layers.Flatten())

    # Camadas Ocultas/Dropout
    net.add(keras.layers.Dense(units=256, activation='relu'))
    net.add(keras.layers.Dropout(0.15))

    net.add(keras.layers.Dense(units=128, activation='relu'))
    net.add(keras.layers.Dropout(0.15))

    # Camada de Saida
    net.add(keras.layers.Dense(5, activation='softmax'))

    # Compilador: Calcula a taxa de perda; a metrica da validacao; otimizacao da fucao de custo usando SGD
    net.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics='accuracy')

    return net

# Controlando a parada do treinamento
stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

'''
# Salvando o melhor modelo
checkpoint_weights = ModelCheckpoint("best_weights_cerranetv3.h5", monitor='loss', verbose=1, save_weights_only=True,
                             save_best_only=True, mode='auto', period=1)

checkpoint_all = ModelCheckpoint("best_model_cerranetv3.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)

'''
# Multiple GPU
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    cerranet = cerranetv3()

# Treinamento
history = cerranet.fit(x_train, epochs=100, callbacks=stopping,
                       validation_data=x_valida)

# Saving last model
cerranet.save_weights('last_cerranetv3_weights.h5')
configs = cerranet.to_json()
with open('best_cerranetv3_json.json', 'w') as json_file:
    json_file.write(configs)

cerranet.save_weights('last_cerranetv3.hdf5')


