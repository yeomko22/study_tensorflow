import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import util
import json
z
filepath = 'sarcasm.json'
sentences = []
labels = []
with open(filepath) as file:
    for i, line in tqdm(enumerate(file), total=util.file_line_count(filepath)):
        line_json = json.loads(line)
        sentences.append(line_json['headline'])
        labels.append(line_json['is_sarcastic'])

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(labels[2], sentences[2])
print(padded[2])
print(padded.shape)
