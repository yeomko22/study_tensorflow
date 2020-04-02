import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
trian_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []


