import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_model():
    local_weights_file = './inception_v3.h5'
    pretrained_model = InceptionV3(input_shape=(150, 150, 3),
                                   include_top=False,
                                   weights=None)
    pretrained_model.load_weights(local_weights_file)

    for layer in pretrained_model.layers:
        layer.trainable = False

    last_layer = pretrained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = Flatten()(last_output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return tf.keras.models.Model(pretrained_model.input, x)


def load_data_generator(base_dir):
    train_dir = os.path.join(base_dir, 'training_set', 'training_set')
    validation_dir = os.path.join(base_dir, 'test_set', 'test_set')
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=10,
                                                        class_mode='binary',
                                                        target_size=(150, 150))
    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            batch_size=10,
                                                            class_mode='binary',
                                                            target_size=(150, 150))
    return train_generator, validation_generator


def plot_accuracy_loss(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.savefig('accuracy.png')

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss.png')


def train():
    model = get_model()
    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    base_dir = './cats_and_dogs'
    train_generator, validation_generator = load_data_generator(base_dir)
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=10,
                                  verbose=1)
    plot_accuracy_loss(history)


if __name__ == '__main__':
    train()

