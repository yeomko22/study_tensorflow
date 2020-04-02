import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt


def get_model():
    model = tf.keras.models.Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=[150, 150, 3]),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model


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
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    base_dir = './cats_and_dogs'
    train_generator, validation_generator = load_data_generator(base_dir)
    history = model.fit_generator(train_generator,
                                  validation_data=validation_generator,
                                  epochs=1,
                                  verbose=1)
    plot_accuracy_loss(history)


if __name__ == '__main__':
    train()

