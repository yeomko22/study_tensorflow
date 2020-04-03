import io

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRUNC_TYPE = 'post'
OOV_TOKEN = "<OOV>"
BUFFER_SIZE = 10000
BATCH_SIZE = 64


def get_data():
    imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']
    tokenizer = info.features['text'].encoder

    train_data = train_data.shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_data))
    test_data = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

    return train_data, test_data, tokenizer


def get_model(tokenizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(tokenizer.vocab_size, EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    train_data, test_data, tokenizer = get_data()
    model = get_model(tokenizer)
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data,
              epochs=10,
              validation_data=test_data)
