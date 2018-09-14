import os
import numpy as np
import tensorflow as tf


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K

import pickle   # for tokenizer dumping


def path_builder(path):
    """ Build path if it does not exist"""
    if not os.path.exists(path):
        os.makedirs(path)
    return


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


def index_word_vectors(filename_to_read, **kwargs): #max_cnt = 100000000):
    max_cnt = kwargs.get("MAX_INDEX_CNT", 1000010000)

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


def vectorize_text(texts, labels, **kwargs):
    """vectorize the text samples into a 2D integer tensor"""
    MAX_NUM_WORDS = kwargs.get("MAX_NUM_WORDS", 200000)
    MAX_SEQUENCE_LENGTH = kwargs.get("MAX_SEQUENCE_LENGTH", 50)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    maxlen = 0
    for i in range(0, len(sequences)):
        maxlen = maxlen if len(sequences[i]) < maxlen else len(sequences[i])

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, word_index, tokenizer


def vectorize_text_test(texts, labels, tokenizer=None, **kwargs):
    """Uses trained tokenizer to transform texts to sequences"""
    MAX_SEQUENCE_LENGTH = kwargs.get("MAX_SEQUENCE_LENGTH", 50)

    if tokenizer==None:
        # loading
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

    sequences = tokenizer.texts_to_sequences(texts)

    maxlen = 0
    for i in range(0, len(sequences)):
        maxlen = maxlen if len(sequences[i]) < maxlen else len(sequences[i])
    if(maxlen > MAX_SEQUENCE_LENGTH):
        print("Found sequence with length {} (words). The maximum allowed length is {}. Script will continue, but your "
              "results may be broken")

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, tokenizer.word_index, tokenizer


def split_data(data, labels, **kwargs):
    """Split the data into validation test and training"""
    VALIDATION_SPLIT = kwargs.get("VALIDATION_SPLIT", 0.2)

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


def prepare_embedding_matrix(word_index, embeddings_index, **kwargs):
    """Preparing embedding matrix to be used in keras mdoel"""
    # parameters
    MAX_NUM_WORDS = kwargs.get("MAX_NUM_WORDS", 200000)
    EMBEDDING_DIM = kwargs.get("EMBEDDING_DIM", 100)

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



def get_data(data_directory, config, tokenizer=None, mode="training"):
    # evaluate trained model on a test set
    txt_files = list_txt_files_in_dir(data_directory)
    texts, target = read_txt_files(txt_files)
    target = np.asarray(target)
    labels = (target == "negative") * 0.0 + (target == "neutral") * 1.0 + (target == "positive") * 2.0

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    if mode == "training":
        data, labels, word_index, tokenizer = vectorize_text(texts, labels, **config)
    elif mode == "test":
        data, labels, word_index, tokenizer = vectorize_text_test(texts=texts, labels=labels, tokenizer=tokenizer, **config)

    return data, labels, word_index, tokenizer

## ---------------------------------------------------------------------------------------------------------------------
#region losses


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

#endregion losses
