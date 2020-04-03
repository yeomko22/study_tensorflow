"""
requirements
1. use real text data in csv format.
2. use pretrained word embedding.
3. use LSTM, apply bidirectional.
4. plot accuracy, loss
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import random
import numpy as np
from tqdm import tqdm

EMBEDDING_DIM = 100
MAX_LENGTH = 16
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"
TEST_PORTION = 0.1
CORPUS_SIZE = 1600000
PRETRAINED_WORDS = 400000


def split_train_test(padded, labels):
    test_size = int(len(padded) * TEST_PORTION)
    train_padded = padded[test_size:]
    test_padded = padded[:test_size]
    train_labels = labels[test_size:]
    test_labels = labels[:test_size]
    return train_padded, train_labels, test_padded, test_labels


def get_data():
    corpus = []
    with open('./sentiment140.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i, line in tqdm(enumerate(csv_reader), total=CORPUS_SIZE):
            corpus.append([line[5], 0 if line[0] == '0' else 1])
    random.shuffle(corpus)

    sentences, labels = ([], [])
    for x in corpus:
        sentences.append(x[0])
        labels.append(x[1])
    labels = np.array(labels)
    print('complete read sentiment140 data')
    return sentences, labels


def tokenize(sentences):
    print('start tokenize sentences')
    tokenizer = Tokenizer(oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(sentences)
    print('complete fit tokenizer on sentences')
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    print('complete tokenize sentences')
    return padded, word_index


def get_pretrained_embedding(vocab_size, word_index):
    print('start loading pretrained word embedding')
    embeddings_index = {}
    with open('glove.6B.100d.txt') as f:
        for i, line in tqdm(enumerate(f), total=PRETRAINED_WORDS):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector
    print('complete read pretrained glov weights')
    return embeddings_matrix


def get_model(vocab_size, embeddings_matrix):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, EMBEDDING_DIM, input_length=MAX_LENGTH, weights=[embeddings_matrix],
                                  trainable=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model


def train(model, train_padded, train_labels, test_data, test_labels):
    history = model.fit(train_padded,
                        train_labels,
                        validation_data=(test_data, test_labels),
                        epochs=10)
    return history


if __name__ == '__main__':
    sentences, labels = get_data()
    padded, word_index = tokenize(sentences)
    vocab_size = len(word_index)
    train_padded, train_labels, test_padded, test_labels = split_train_test(padded, labels)
    embeddings_matrix = get_pretrained_embedding(vocab_size, word_index)
    model = get_model(vocab_size, embeddings_matrix)
    history = train(model, train_padded, train_labels, test_padded, test_labels)
