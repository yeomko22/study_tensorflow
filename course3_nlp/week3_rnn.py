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


def get_data():
    imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    training_labels = []

    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(str(s.numpy()))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(str(s.numpy()))
        testing_labels.append(l.numpy())

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    print(len(training_labels_final), len(testing_labels_final))
    return training_sentences, training_labels_final, testing_sentences, testing_labels_final


def tokenize(training_sentences, testing_sentences):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(training_sentences)
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LENGTH)
    return training_padded, testing_padded, tokenizer.word_index


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def write_embedding(model, reverse_word_index):
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    embedding_layer = model.layers[0]
    weights = embedding_layer.get_weights()[0]
    for word_num in range(1, VOCAB_SIZE):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()


if __name__ == '__main__':
    training_sentences, training_labels, testing_sentences, testing_labels = get_data()
    training_padded, testing_padded, word_index = tokenize(training_sentences, testing_sentences)
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    model = get_model()
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_padded,
              training_labels,
              epochs=10,
              validation_data=(testing_padded, testing_labels))
    write_embedding(model, reverse_word_index)
