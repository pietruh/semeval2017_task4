"""Main file with loading data and training RNN for sentiment twitter analysis"""

import nltk
import keras
import gensim
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, MaxoutDense, Dense, Activation

from keras.constraints import maxnorm
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam

import numpy as np

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
import pickle
import timeit



def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def get_RNN_model(embeddings):
    """First approach to model with Recurrent Units - LSTMs. This may be very dirty and hacky"""
    classes = 3
    max_length = 50
    masking = True
    return_sequences = True
    consume_less = 'cpu'
    dropout_U = 0.3
    model = Sequential()
    dropout_rnn = 0.3
    dropout_final = 0.5
    loss_l2 = 0.0001
    clipnorm = 1.
    lr = 0.001
    # define embedding input layer

    # TODO: This have to be worked-around as shown in https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    embedding = Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        input_length=max_length if max_length > 0 else None,
        trainable=False,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )
    model.add(embedding)

    rnn_layer_orig = get_RNN()
    model.add(rnn_layer_orig)

    # define bidirectional LSTM layer no. 1

    # Add dropout for regularization after LSTM layer no. 1
    model.add(Dropout(dropout_rnn))

    # define bidirectional LSTM layer no. 2
    rnn_layer_orig_2 = get_RNN()

    # model.add(rnn_layer_orig_2)
    # # define bidirectional LSTM layer no. 2
    # rnn_2 = Bidirectional(LSTM(64, return_sequences=return_sequences,
    #                            consume_less=consume_less, dropout_U=dropout_U,
    #                            W_regularizer=l2(0.)))
    # model.add(rnn_2)
    # Add dropout for regularization after LSTM layer no. 2
    model.add(Dropout(dropout_rnn))

    # model.add(MaxoutDense(100, input_dim=(None, 50), W_constraint=maxnorm(2)))
    #
    # #TODO(MP): Until this it should work
    # model.add(Dropout(dropout_final))


    # define bidirectional LSTM layer no. 1

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy')
    return model

def get_embedding_matrix(model):
    """convert the wv word vectors into a numpy matrix that is suitable for insertion
    into our TensorFlow and Keras models
    """
    embedding_matrix = np.zeros((model.vectors.shape[0], model.vectors.shape[1]))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def list_txt_files_in_dir(directory):
    """List .txt files in the given directory"""
    import os
    list_files = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            list_files.append(os.path.join(directory, file))
    return list_files


def tokenize_txt_files(list_files, tokenizer):
    """Given list of files with tweets, parse and tokenize it. Returning x, y."""
    # load data
    data = []
    target = []
    tokenized_1 = []
    for file_name_txt in list_files:

        with open(file_name_txt, 'r') as file:
            for line in file:
                splitted = line.split("\t")
                target.append(splitted[1])
                data.append(splitted[2])
                tokenized_1.append(tokenizer.tokenize(splitted[2]))
        print("Total {} lines after reading from file {}".format(len(tokenized_1), file_name_txt))
    print("Read {} lines from files {}".format(len(tokenized_1), list_files))

    return tokenized_1, target



# region main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    glove_data_file_w2v = r'./data/glove.twitter.27B/glove_twitter_27B_25d_w2vformat.txt'
    sentiment_train_file = r'./data/sentiment_train/twitter-2016train-A.txt'
    sentiment_test_file = r'./data/sentiment_train/twitter-2016test-A.txt'

    train_directory = r'./data/sentiment_train/'
    test_directory = r'./data/sentiment_test/'
    train_txt_files = list_txt_files_in_dir(train_directory)
    # to convert gloVe embeddings to word2vec use this
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove2word2vec(glove_data_file, glove_data_file_w2v)


    # tokenize data
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

    tokenized_1, target = tokenize_txt_files(train_txt_files, tokenizer)

    # load embeddings
    # TODO: measure time



    start_load_glove = timeit.timeit()
    import gensim.downloader as api

    info = api.info()  # show info about available models/datasets
    model_emb = api.load("glove-twitter-25")  # download the model and return as object ready for use
    model_emb.most_similar("cat")
    embedding_matrix = get_embedding_matrix(model_emb)
    # model_emb = gensim.models.KeyedVectors.load_word2vec_format(glove_data_file_w2v, binary=False)
    end_load_glove = timeit.timeit()
    print(end_load_glove - start_load_glove)
    model_emb.similarity("later", "sooner")

    # get model
    RNN_model = get_RNN_model(embedding_matrix)

    cut_set = 4500
    tokenized_1_train = tokenized_1[:cut_set]
    tokenized_1_test = tokenized_1[cut_set:]

    target_train = target[:cut_set]
    target_test = target[cut_set:]


    #TODO(MP): work on that fit - use generators etc.
    history = RNN_model.fit(x=tokenized_1_train, y=target_train, batch_size=50, epochs=1, verbose=1, callbacks=None,
                  validation_split=0.2,
    # validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    # validation_steps=None
                  )

    pickle.dump(history.history,
                open("hist_task4_subA_mike.pickle", "wb"))

    RNN_model.save(filepath="RNN_model.h5", overwrite=True)
    results = RNN_model.evaluate(x=tokenized_1_test, y=target_test, batch_size=50, verbose=1
                       #, sample_weight=None, steps=None
    )
    print(results)
# ----------------------------------------------------------------------------------------------------------------------
# endregion main


#TODO(MP): use this: http://adventuresinmachinelearning.com/gensim-word2vec-tutorial/ to get closest words?
