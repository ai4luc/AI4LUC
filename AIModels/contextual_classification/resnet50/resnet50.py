"""
ResNet50 adapted from Keras Applications
https://keras.io/api/applications/#usage-examples-for-image-classification-models
by Mateus de Souza Miranda
2023
"""

# Pack
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


# Set up
path_trained_models = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/AIModels/trained_models/resnet50/'
path_datatrain = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/train2/images/'

# Data
img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Separando dados para as variaveis de treinamento e validacao
x_train = img.flow_from_directory(path_datatrain,
                                  target_size=(256, 256),
                                  batch_size=128,
                                  color_mode='rgb',
                                  class_mode="categorical",
                                  subset='training')

# Model
resnet50 = ResNet50(weights=None, include_top=False, input_shape=(256, 256, 3))

# Add a global spatial average pooling layer
x = resnet50.output
x = GlobalAveragePooling2D()(x)
# Let's add a fully-connected layer
x = Dense(256, activation='relu')(x)
# And a logistic layer, have 8 classes
predictions = Dense(8, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=resnet50.input, outputs=predictions)

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='accuracy')


# Stopping training
stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=8)

# Save the best model per epoch
checkpoint_weights = ModelCheckpoint(path_trained_models+"best_tweights_resnet50.h5", monitor='loss',
                                     verbose=1, save_weights_only=True,
                                     save_best_only=True, mode='auto', save_freq=1)

checkpoint_all = ModelCheckpoint(path_trained_models+"best_tmodel_resnet50.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', save_freq=1)

print(model.summary())

# Training
history = model.fit(x_train, epochs=80, callbacks=[stopping, checkpoint_all])

# Saving the last ran model
model.save_weights(path_trained_models+'resnet50t_weights.h5')
configs = model.to_json()
with open(path_trained_models+'resnet50t_json.json', 'w') as json_file:
    json_file.write(configs)

model.save_weights(path_trained_models+'resnet50t.hdf5')

