"""
Main file with loading data and training RNN for sentiment twitter analysis
"""

from __future__ import print_function

import os
import sys
from keras.layers import Embedding
from keras.models import load_model
from keras.initializers import Constant
from models import get_RNN_model_w_layer
from nlp_utilities import path_builder, index_word_vectors, get_data, split_data, prepare_embedding_matrix, \
    macro_averaged_recall_tf_onehot, \
    macro_averaged_recall_tf_soft, gpu_configuration_initialization

from config import config, model_config, DEBUG

# Import callbacks that will be passed to the fit functions
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Import custom objects that were created for this task
from logger_to_file import Logger
from data_generator import SynonymDataGenerator
from keras.utils.generic_utils import get_custom_objects

# ----------------------------------------------------------------------------------------------------------------------
# region main

if __name__ == "__main__":
    # Set up environmental variables etc.
    continue_training = 0
    # Name of the training session to which all objects will be saved
    NAME_OF_THE_TRAIN_SESSION = "8_testing_augmented_data_full_model+gaussian+maxout+macro_loss_+full_data"
    PATH_TO_THE_LEARNING_SESSION = "./learning_sessions/" + NAME_OF_THE_TRAIN_SESSION + "/"

    # Name of the pretrained model, if continue_training=1
    pretrained_filepath = PATH_TO_THE_LEARNING_SESSION + "rnn_model_3.h5"

    print("DEBUG = {} \nconfig = {}\nmodel_config = {}".format(DEBUG, config, model_config))

    # Update custom object with my own loss functions
    custom_objects = {'macro_averaged_recall_tf_onehot': macro_averaged_recall_tf_onehot,
                      'macro_averaged_recall_tf_soft': macro_averaged_recall_tf_soft
                      }
    get_custom_objects().update(custom_objects)

    # Configuring gpu for the training
    gpu_configuration_initialization()
    use_generator = 0

    # Build directories
    path_builder(PATH_TO_THE_LEARNING_SESSION)
    # Create logger to files (copying std out stream to a file)
    sys.stdout = Logger(PATH_TO_THE_LEARNING_SESSION + "log_training")

    # First, build index mapping words in the embeddings set to their embedding vector
    print("Indexing word vectors.")
    # TODO: Check datastories.twitter dataset and their preprocessing toolkit ekphrasis
    # Using twitter specific dataset. Pretrained by glove.
    filename_to_read = os.path.join("./data/glove.twitter.27B/", "glove.twitter.27B.200d.txt")
    embeddings_index = index_word_vectors(filename_to_read, **config)
    print("Found {} word vectors.".format(len(embeddings_index)))

    # Second, prepare text samples and their labels
    print("Processing text dataset")

    train_directory = r'./data/senti_short/sentiment_train/' #r'./data/sentiment_train/'
    test_directory = r'./data/senti_short/sentiment_test/' #r'./data/sentiment_test/'

    # Third, vectorize the training text samples into a 2D integer tensor
    data, labels, word_index, tokenizer, texts = get_data(train_directory, config, tokenizer=None, mode="training")

    # split the data into a training set and a validation set
    x_train, y_train, x_val, y_val = split_data(data, labels, **config)

    # Fouth, vectorize the testing set using provided tokenizer
    test_data, test_labels, test_word_index, test_tokenizer, test_texts = get_data(test_directory, config,
                                                                                   tokenizer=tokenizer,
                                                                                   mode="test")

    # Fifth, prepare embedding matrix
    embedding_matrix, num_words = prepare_embedding_matrix(word_index, embeddings_index, **config)

    # Sixth, load pre-trained word embeddings into an Embedding layer. Additionally- keep the embeddings fixed.
    embedding_layer = Embedding(num_words,
                                config["EMBEDDING_DIM"],
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=config["MAX_SEQUENCE_LENGTH"],
                                trainable=False)

    # Seventh, build the model with the use of the Embedding layer
    model = get_RNN_model_w_layer(embedding_layer, macro_averaged_recall_tf_onehot, macro_averaged_recall_tf_soft)

    # If there is a desire to continue training of a model, this may be used
    if continue_training:
        model = load_model(pretrained_filepath)

        print("Successfully loaded pretrained model {}".format(pretrained_filepath))

    # Eighth, set callbacks
    callbacks = []

    # Create model checkpointer to save best models during training
    model_checkpoint = ModelCheckpoint(PATH_TO_THE_LEARNING_SESSION + 'model_ckpt.h5',
                                       monitor='val_loss',
                                       save_weights_only=False,
                                       save_best_only=True)
    callbacks.append(model_checkpoint)

    # Callback for stopping the training if no progress is achieved
    early_stopper = EarlyStopping(monitor='val_loss', patience=2)
    callbacks.append(early_stopper)

    # Ninth, run training. Either fit_generator that augments data using synonymization of a choosen words or fit in
    # a regular way

    # Use data generator that augments twitter messages online, during the training
    if use_generator:
        synonym_train_generator = SynonymDataGenerator(batch_size=config["BATCH_SIZE"], train_data_x=data,
                                                       train_data_y=labels,
                                                       embeddings_index=embeddings_index, word_index=word_index,
                                                       config=config)
        print("Created DataGenerator object")
        print("Fitting generator")

        hitory_generator = model.fit_generator(generator=synonym_train_generator,
                                               epochs=config["EPOCHS"],
                                               steps_per_epoch=config["BATCH_SIZE"],
                                               validation_data=(x_val, y_val),
                                               callbacks=callbacks
                                               )
        print("Fitting generator finished!")
        print("Evaluating generator!")
        loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])
        print("Generator Evaluated !")


    else:
        # Do not use data augmentation during the training
        print("Fitting normal way ...1")
        history = model.fit(data, labels,
                            batch_size=config["BATCH_SIZE"],
                            epochs=10,
                            validation_data=(test_data, test_labels),
                            callbacks=callbacks)

        print("Fitting normal way Finished...!")
        model.save(PATH_TO_THE_LEARNING_SESSION + "rnn_model_1_cont.h5")

        loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])
        print("Evaluated metrics = {} \n {}".format(loss_metrics_eval, model.metrics_names))

        print("Fitting normal way ...2!")
        history = model.fit(data, labels,
                            batch_size=config["BATCH_SIZE"],
                            epochs=5,
                            validation_data=(test_data, test_labels),
                            callbacks=callbacks)
        print("Fitting normal way Finished...!")
        model.save(PATH_TO_THE_LEARNING_SESSION + "rnn_model_2_cont.h5")

        loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])
        print("Evaluated metrics = {} \n {}".format(loss_metrics_eval, model.metrics_names))

        print("Fitting normal way ...3!")
        history = model.fit(data, labels,
                            batch_size=config["BATCH_SIZE"],
                            epochs=5,
                            validation_data=(test_data, test_labels),
                            callbacks=callbacks)
        print("Fitting normal way Finished...!")
        model.save(PATH_TO_THE_LEARNING_SESSION + "rnn_model_3_cont.h5")

        loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])
        print("Evaluated metrics = {} \n {}".format(loss_metrics_eval, model.metrics_names))

    print("Script finished successfully!")
# endregion main
