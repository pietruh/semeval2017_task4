"""
Utilities functions for setting up environment, data processing pipeline, saving results to file and losses
definition
"""

import os
import numpy as np
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K

import pickle  # for tokenizer dumping
from text_processor import text_processor

## ---------------------------------------------------------------------------------------------------------------------
# region utilities
def gpu_configuration_initialization():
    """
    Set Keras GPU memory allocation mode.
    """

    GPU_config = tf.ConfigProto()
    GPU_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=GPU_config))
    return


def path_builder(path):
    """Build path if it does not exist
    :param path: Path to the directory (if it doesn't exist it will be created)"""
    if not os.path.exists(path):
        os.makedirs(path)
    return


def remove_duplicates(texts):
    """
    Remove possible duplicates from list
    :param texts: list with training tweets
    :type texts: list of str
    """
    return list(set(texts))


def list_txt_files_in_dir(directory):
    """
    List .txt files in the given directory
    :param directory: Path to the directory to be checked
    :type directory: str
    :returns: List of .txt files in the directory
    :rtype: list of str
    """
    import os
    list_files = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            list_files.append(os.path.join(directory, file))
    return list_files


def read_txt_files(list_files):
    """
    Given list of files with tweets, parse and tokenize it. Returning text data and target as text
    :param list_files: List of the filenames that contains tweeter data that is complaint with Semeval2017-Task 4 A
    :type list_files: list of str
    :returns: data - texts of tweets, target - it's classification
    :rtype data: list of str
    :rtype target: list of str
    """

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


def index_word_vectors(filename_to_read, **kwargs):
    """
    Creating dictionary that maps word to ID based on some word2vec pretrained file. It will be used in the embedding layer.
    :param filename_to_read: Name of the file that contains word2vec word embeddings.
    :param kwargs: Configuration dictionary. Expected to see MAX_INDEX_CNT
    :return embeddings_index: dictionary that assigns vector coefficients to the word
    """
    max_cnt = kwargs.get("MAX_INDEX_CNT", 1000010000)

    embeddings_index = {}
    with open(filename_to_read, encoding="utf8") as f:
        cnt = 0
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            cnt += 1
            if cnt > max_cnt:
                break
    return embeddings_index


def vectorize_text(texts, labels, **kwargs):
    """
    Vectorize the text samples into a 2D integer tensor.
    :param texts: Text that will be used by the tokenizer for fitting
    :param labels: Classified labels will be converted to categorical
    :param MAX_NUM_WORDS:
    :param MAX_SEQUENCE_LENGTH:
    :param kwargs: Expected to see
    :returns data: Padded text converted to IDs
    :returns labels: Labels as categorical, one-hot encoded vector
    :returns word_index: Look-up table to be used in Embedding layer
    :returns tokenizer: Tokenizer, it will be used to prepare test dataset
    """

    MAX_NUM_WORDS = kwargs.get("MAX_NUM_WORDS", 200000)
    MAX_SEQUENCE_LENGTH = kwargs.get("MAX_SEQUENCE_LENGTH", 100)
    TEXT_PREPROCESSING = kwargs.get("TEXT_PREPROCESSING", 1)

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    processed_text = []
    if TEXT_PREPROCESSING:
        for text in texts:
            processed_text.append(text_processor.pre_process_doc(text))

    tokenizer.fit_on_texts(processed_text)
    sequences = tokenizer.texts_to_sequences(processed_text)
    word_index = tokenizer.word_index
    print("Found {} unique tokens.".format(len(word_index)))

    # saving tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    maxlen = 0
    for i in range(0, len(sequences)):
        maxlen = maxlen if len(sequences[i]) < maxlen else len(sequences[i])

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(labels)
    print("Shape of data tensor: {}".format(data.shape))
    print("Shape of label tensor: {}".format(labels.shape))
    return data, labels, word_index, tokenizer


def vectorize_text_test(texts, labels, tokenizer=None, **kwargs):
    """
    Uses trained tokenizer to transform texts to sequences
    :param texts: List of strings that will be turned into vectors.
    :param labels: Labels will be turned into one-hot encoded targets.
    :param tokenizer: Either provide a tokenizer or it will be loaded from default hardcoded name
    :param kwargs: Aditional arguments that may define MAX_SEQUENCE_LENGTH and TOKENIZER_PATH
    :returns data: Padded text converted to IDs
    :returns labels: Labels as categorical, one-hot encoded vector
    :returns word_index: Look-up table to be used in Embedding layer.
    :returns tokenizer: Tokenizer, it will be used to prepare test dataset
    :rtype: object
    """

    MAX_SEQUENCE_LENGTH = kwargs.get("MAX_SEQUENCE_LENGTH", 100)
    TOKENIZER_PATH = kwargs.get("TOKENIZER_PATH", "./tokenizer.pickle")
    TEXT_PREPROCESSING = kwargs.get("TEXT_PREPROCESSING", 1)

    if tokenizer == None:
        # loading
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)

    processed_text = []
    if TEXT_PREPROCESSING:
        for text in texts:
            processed_text.append(text_processor.pre_process_doc(text))

    sequences = tokenizer.texts_to_sequences(processed_text)

    maxlen = 0
    for i in range(0, len(sequences)):
        maxlen = maxlen if len(sequences[i]) < maxlen else len(sequences[i])
    if (maxlen > MAX_SEQUENCE_LENGTH):
        print("Found sequence with length {} (words). The maximum allowed length is {}. Script will continue, but your "
              "results may be broken")

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    return data, labels, tokenizer.word_index, tokenizer


def split_data(data, labels, **kwargs):
    """
    Split the data into validation test and training
    :param data: Network input, called x
    :param labels: Network target, called y
    :param kwargs: VALIDATION_SPLIT can be defined as keyword argument
    :returns x_train, y_train, x_val, y_val: Splitted dataset for the training purposes
    """

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
    """
    Preparing embedding matrix to be used in keras model
    :param word_index: dictionary mapping words to index
    :param embeddings_index: dictionary that assigns vector coefficients to the word
    :param kwargs: Configuration dictionary, expecting to see values of MAX_NUM_WORDS, EMBEDDING_DIM
    :return embedding_matrix: Look-up Table that for each entry(word id) gets vector coefficients
    :return num_words: Number of words that the matrix can keep
    """
    # parameters
    MAX_NUM_WORDS = kwargs.get("MAX_NUM_WORDS", 200000)
    EMBEDDING_DIM = kwargs.get("EMBEDDING_DIM", 100)

    print("Preparing embedding matrix.")

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
    """
    Load the data and return for training. This will search given directory looking for data, read and parse files and
    vectorize text.
    :param data_directory: Directory in which there are .txt files with twitter messages
    :param config: Configuration dictionary
    :param tokenizer: Tokenizer is mandatory for testing(it can be loaded from default if not given), and will be
                      returned if training
    :param mode: Either fitting tokenizer on training data or using it to tokenize test data
    :return data: Vectorized data,
    :return labels: one-hot encoded labels,
    :return word_index:  dictionary mapping words to index
    :return tokenizer: tokenizer that was used
    """
    txt_files = list_txt_files_in_dir(data_directory)
    texts, target = read_txt_files(txt_files)
    target = np.asarray(target)
    labels = (target == "negative") * 0.0 + (target == "neutral") * 1.0 + (target == "positive") * 2.0

    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    if mode == "training":
        data, labels, word_index, tokenizer = vectorize_text(texts, labels, **config)
    elif mode == "test":
        data, labels, word_index, tokenizer = vectorize_text_test(texts=texts, labels=labels, tokenizer=tokenizer,
                                                                  **config)

    return data, labels, word_index, tokenizer, texts


def save_predictions(prediction_directory, predictions):
    """
    Saving predictions with their probabilities
    :param prediction_directory: Directory in which file will be saved
    :param predictions: Arrays with predictions. For each tweet there is an 3 cells array with Negative, Neutral,
                        Positive probabilities
    """
    with open(prediction_directory, "w") as text_file:
        for i in range(0, len(predictions)):
            text_file.write(
                "Negative = {}, Neutral = {}, Positive = {}\n".format(predictions[i][0], predictions[i][1],
                                                                      predictions[i][2]))
    return


def save_text_as_tweet(x, y, filename):
    """
    Saving predictions in the original format, with ID, class label and twitter message
    :param x: Texts of the twitter messages
    :param y: Arrays with predictions. For each tweet there is an 3 cells array with Negative, Neutral,
              Positive probabilities
    :param filename: Filename in the directory in which file will be saved
    """
    id = 0
    with open(filename, 'w') as handle:
        for line in range(0, len(x)):
            arg = np.argmax(y[id])
            label = int(arg == 0) * "negative" + int(arg == 1) * "neutral" + int(arg == 2) * "positive"
            handle.write("{}\t{}\t{}\n".format(id, label, x[id]))
            id += 1
    print("Saved {} lines into file {}".format(id, filename))
    return
# endregion utilities
## ---------------------------------------------------------------------------------------------------------------------

## ---------------------------------------------------------------------------------------------------------------------
# region losses


def recall(y_true, y_pred):
    """Recall metric, for one class
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    :param y_true:
    :param y_pred:
    :return:
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def recall_soft(y_true, y_pred):
    """
    Recall metric, for one class, without rounding.
    It can be used as a loss metrics as it's gradient can be calculated.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    :param y_true: Ground-truth values for classification
    :param y_pred: Predicted values of classifications
    :return: recall = Correctly predicted with that label/ #All ground-truth values with that label
    """

    true_positives = K.sum(y_true * y_pred)
    possible_positives = K.sum(y_true)
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def macro_averaged_recall_tf_soft(y_true, y_pred):
    """
    Using softmax output to macro averaging - averaging over classes. Calculate per class recall and average.
    This is a differentiable loss function.
     :param y_true: Ground-truth values for classification
    :param y_pred: Predicted values of classifications
    :return: Recall averaged across classes
    """
    r_neg = recall_soft(y_true[:, 0], y_pred[:, 0])
    r_neu = recall_soft(y_true[:, 1], y_pred[:, 1])
    r_pos = recall_soft(y_true[:, 2], y_pred[:, 2])
    # return as negative so it will be minimized and absolute value of recall with be maximized
    return -(r_neg + r_neu + r_pos) / 3


def macro_averaged_recall_tf_onehot(y_true, y_pred):
    """
    Using one hot encoded output to macro averaging - averaging over classes. Calculate per class recall and average.
    Due to rounding and argmax ops it is not differentiable.
     :param y_true: Ground-truth values for classification
    :param y_pred: Predicted values of classifications
    :return: recall = Correctly predicted with that label/ #All ground-truth values with that label
    """
    max_vals = tf.argmax(y_pred, axis=1)
    y_pred_one_hot = tf.one_hot(max_vals, depth=3)
    r_neg = recall(y_true[:, 0], y_pred_one_hot[:, 0])
    r_neu = recall(y_true[:, 1], y_pred_one_hot[:, 1])
    r_pos = recall(y_true[:, 2], y_pred_one_hot[:, 2])
    # return as negative so it will be minimized and absolute value of recall with be maximized
    return -(r_neg + r_neu + r_pos) / 3

# endregion losses
