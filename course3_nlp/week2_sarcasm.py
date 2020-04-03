import io

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import util
import json

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 32
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'
OOV_TOKEN = "<OOV>"
TRAINING_SIZE = 20000


def get_data(filepath):
    sentences = []
    labels = []
    with open(filepath) as file:
        for i, line in tqdm(enumerate(file), total=util.file_line_count(filepath)):
            line_json = json.loads(line)
            sentences.append(line_json['headline'])
            labels.append(line_json['is_sarcastic'])
    training_sentences = sentences[:TRAINING_SIZE]
    training_labels = labels[:TRAINING_SIZE]
    testing_sentences = sentences[TRAINING_SIZE:]
    testing_labels = labels[TRAINING_SIZE:]
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)
    return training_sentences, training_labels_final, testing_sentences, testing_labels_final


def tokenize(training_sentences, testing_sentences):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    training_sequence = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequence, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE, padding=PADDING_TYPE)
    testing_sequence = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequence, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    return training_padded, testing_padded, word_index


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        tf.keras.layers.GlobalAveragePooling1D(),
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
    filepath = './sarcasm/sarcasm.json'
    training_sentences, training_labels, testing_sentences, testing_labels = get_data(filepath)
    training_padded, testing_padded, word_index = tokenize(training_sentences, testing_sentences)
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    model = get_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_padded,
              training_labels,
              epochs=10,
              validation_data=(testing_padded, testing_labels))
    write_embedding(model, reverse_word_index)
