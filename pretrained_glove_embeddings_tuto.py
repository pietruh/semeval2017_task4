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
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import tensorflow as tf
import keras.backend as K

import keras

from models import get_RNN_model_w_layer
from nlp_utilities import path_builder, index_word_vectors, get_data, split_data, prepare_embedding_matrix, \
    macro_averaged_recall_tf, \
    macro_averaged_recall_tf_soft

# Import callbacks that will be passed to the fit functions
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler

from logger_to_file import Logger
from data_generator import SynonymDataGenerator

NAME_OF_THE_TRAIN_SESSION = "1_test_session"  # name of the training session to which all objects will be saved
PATH_TO_THE_LEARNING_SESSION = "./learning_sessions/" + NAME_OF_THE_TRAIN_SESSION + "/"



# region main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    path_builder(PATH_TO_THE_LEARNING_SESSION)
    sys.stdout = Logger(PATH_TO_THE_LEARNING_SESSION + "log_training")

    model_config = {"classes": 3,
                    "max_length": 50,
                    "masking": True,
                    "return_sequences": True,
                    "consume_less": 'cpu',
                    "dropout_U": 0.3,
                    "dropout_rnn": 0.3,
                    "dropout_final": 0.5,
                    "loss_l2": 0.0001,
                    "clipnorm": 5.,
                    "lr": 0.001,
                    }

    config = {"GLOVE_DIR": 'glove.6B',
              "MAX_SEQUENCE_LENGTH": 50,
              "OUTPUT_CLASSES": 3,
              "MAX_NUM_WORDS": 200000,
              "MAX_INDEX_CNT": 1000,  # 010000,
              "EMBEDDING_DIM": 100,
              "VALIDATION_SPLIT": 0.2,
              "SYNONIMIZE_FRACTION": 0.2,
              "SYNONIMIZE_WORDS": 4,
              "SYNONIMIZE_WORDS_FRACTION": 0.1,
              "model_config": model_config,
              "BATCH_SIZE": 128,
              "EPOCHS": 3
              }

    print('Indexing word vectors.')
    filename_to_read = os.path.join(config["GLOVE_DIR"], 'glove.6B.100d.txt')
    embeddings_index = index_word_vectors(filename_to_read, **config)
    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    train_directory = r'./data/sentiment_train/'
    test_directory = r'./data/sentiment_test/'
    print("Inspect results")

    # finally, vectorize the text samples into a 2D integer tensor
    data, labels, word_index, tokenizer = get_data(train_directory, config, tokenizer=None, mode="training")

    # split the data into a training set and a validation set
    x_train, y_train, x_val, y_val = split_data(data, labels, **config)

    test_data, test_labels, test_word_index, test_tokenizer = get_data(test_directory, config, tokenizer=tokenizer,
                                                                       mode="test")

    # prepare embedding matrix
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index, **config)


    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                config["EMBEDDING_DIM"],
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=config["MAX_SEQUENCE_LENGTH"],
                                trainable=False)

    print('Training model.')
    rnn_model = get_RNN_model_w_layer(embedding_layer, macro_averaged_recall_tf, macro_averaged_recall_tf_soft)
    model = rnn_model

    y_pred = model.predict(x_train)

    # set callbacks
    callbacks = []

    ## Callbacks - setup your callbacks for training generator
    # Create model checkpointer to save best models during training
    model_checkpoint = ModelCheckpoint('weights_ckpt.hdf5',
                                            monitor='val_macro_averaged_recall_tf',
                                            save_best_only=True)
    callbacks.append(model_checkpoint)

    # Callback for stopping the training if no progress is achieved
    early_stopper = EarlyStopping(monitor='val_macro_averaged_recall_tf', min_delta=0, patience=0, verbose=0,
                                  mode='auto', baseline=None)
    callbacks.append(early_stopper)

    # Changing learning rate
    #lr_scheduler = LearningRateScheduler(schedule, verbose=0)

    synonym_train_generator = SynonymDataGenerator(batch_size=config["BATCH_SIZE"], train_data_x=data,
                                                   train_data_y=labels,
                                                   embeddings_index=embeddings_index, word_index=word_index,
                                                   config=config)


    hitory_generator = model.fit_generator(generator=synonym_train_generator,
                                           batch_size=config["BATCH_SIZE"],
                                           epochs=config["EPOCHS"],
                                           validation_data=(x_val, y_val),
                                           callbacks=callbacks
                                           )

    history = model.fit(x_train, y_train,
                        batch_size=config["BATCH_SIZE"],
                        epochs=config["EPOCHS"],
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)

    model.save("rnn_model_b128_ep3_1_maxindex.h5")


    loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])
    print("Evaluated metrics = {} \n {}".format(loss_metrics_eval, model.metrics_names))

