'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import tensorflow as tf
import keras.backend as K

import keras

from models import get_RNN_model_w_layer

# Import callbacks that will be passed to the fit functions
from keras.callbacks import ModelCheckpoint, TensorBoard

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 50
MAX_NUM_WORDS = 200000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def list_txt_files_in_dir(directory):
    """List .txt files in the given directory"""
    import os
    list_files = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            list_files.append(os.path.join(directory, file))
    return list_files


def read_txt_files(list_files):
    """Given list of files with tweets, parse and tokenize it. Returning x, y."""
    # load data
    data = []
    target = []
    for file_name_txt in list_files:
        with open(file_name_txt, 'r') as file:
            for line in file:
                splitted = line.split("\t")
                target.append(splitted[1])
                data.append(splitted[2])
        print("Total {} lines after reading from file {}".format(len(data), file_name_txt))
    print("Read {} lines from files {}".format(len(data), list_files))
    return data, target


def index_word_vectors(filename_to_read, max_cnt = 100000000):
    embeddings_index = {}
    with open(filename_to_read, encoding="utf8") as f:
        cnt = 0
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            cnt+=1
            if cnt > max_cnt:
                break
    return embeddings_index


def vectorize_text(texts, labels):
    """vectorize the text samples into a 2D integer tensor"""
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    maxlen = 0
    for i in range(0, len(sequences)):
        maxlen = maxlen if len(sequences[i]) < maxlen else len(sequences[i])

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, word_index


def split_data(data, labels):
    """Split the data into validation test and training"""
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    return x_train, y_train, x_val, y_val


def prepare_embedding_matrix(word_index):
    """Preparing embedding matrix to be used in keras mdoel"""
    print('Preparing embedding matrix.')

    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix, num_words


def recall(y_true, y_pred):
    """Recall metric, for one class
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def recall_soft(y_true, y_pred):
    """Recall metric, for one class, without rounding.
    It can be used as a loss metrics as it's gradient can be calculated
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """

    true_positives = K.sum(y_true * y_pred)
    possible_positives = K.sum(y_true)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def macro_averaged_recall_tf(y_true, y_pred):
    """Macro averaging - averaging over classes. Calculate per class recall and average."""
    r_neg = recall_soft(y_true[:, 0], y_pred[:, 0])
    r_neu = recall_soft(y_true[:, 1], y_pred[:, 1])
    r_pos = recall_soft(y_true[:, 2], y_pred[:, 2])
    return (r_neg + r_neu + r_pos) / 3


def macro_averaged_recall_tf_soft(y_true, y_pred):
    """Macro averaging - averaging over classes. Calculate per class recall and average."""
    max_vals = tf.argmax(y_pred, axis=1)
    y_pred_one_hot = tf.one_hot(max_vals, depth=3)
    r_neg = recall(y_true[:, 0], y_pred_one_hot[:, 0])
    r_neu = recall(y_true[:, 1], y_pred_one_hot[:, 1])
    r_pos = recall(y_true[:, 2], y_pred_one_hot[:, 2])
    return (r_neg + r_neu + r_pos) / 3


def get_data(test_directory):
    # evaluate trained model on a test set
    test_txt_files = list_txt_files_in_dir(test_directory)
    test_texts, test_target = read_txt_files(test_txt_files)
    test_target = np.asarray(test_target)
    test_labels = (test_target == "negative") * 0.0 + (test_target == "neutral") * 1.0 + (
                                                                                         test_target == "positive") * 2.0

    print('Found %s texts.' % len(test_texts))

    # finally, vectorize the text samples into a 2D integer tensor
    test_data, test_labels, test_word_index = vectorize_text(test_texts, test_labels)
    return test_data, test_labels, test_word_index


# region main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # first, build index mapping words in the embeddings set
    # to their embedding vector

    print('Indexing word vectors.')
    filename_to_read = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
    embeddings_index = index_word_vectors(filename_to_read, max_cnt = 1000010000) # set this to 1000 only for debugging
    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    train_directory = r'./data/sentiment_train/'
    test_directory = r'./data/sentiment_test/'
    test_data, test_labels, test_word_index = get_data(test_directory)

    train_txt_files = list_txt_files_in_dir(train_directory)
    texts, target = read_txt_files(train_txt_files)
    target = np.asarray(target)
    labels = (target == "negative")*0.0 + (target == "neutral")*1.0 + (target == "positive")*2.0

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    data, labels, word_index = vectorize_text(texts, labels)

    # split the data into a training set and a validation set
    x_train, y_train, x_val, y_val = split_data(data, labels)

    # prepare embedding matrix
    embedding_matrix, num_words = prepare_embedding_matrix(word_index)


    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')
    rnn_model = get_RNN_model_w_layer(embedding_layer, macro_averaged_recall_tf, macro_averaged_recall_tf_soft)
    model = rnn_model


    y_pred = model.predict(x_train)

    # set callbacks
    callbacks = []

    # Callbacks - setup your callbacks for training generator
    model_checkpoint = ModelCheckpoint('weights_ckpt.hdf5',
                                            monitor='val_macro_averaged_recall_tf',
                                            save_best_only=True)
    callbacks.append(model_checkpoint)

    history = model.fit(x_train, y_train,
              batch_size=128,
              epochs=20,
              validation_data=(x_val, y_val),
                        callbacks=callbacks)

    model.save("rnn_model_b128_ep20_4.h5")



    loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=128)
    print("Evaluated metrics = {} \n {}".format(loss_metrics_eval, model.metrics_names))

