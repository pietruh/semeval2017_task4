"""This file contains configuration dictionaries for training, testing and offline data generation. Some of this settings may be overriden in the specific part of the offline data generation"""
DEBUG = 0
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

config = {
    "MODEL_NAME": "rnn_model.h5",  # Name of the model, to be saved in the train_model.py
    # and that will be read in the test_model
    "NAME_OF_THE_TRAIN_SESSION": "3_still_testing",
    # Name of the training session
    "TRAIN_DIRECTORY": r'./data/sentiment_train/',  # Directory with the trainingdata
    "TEST_DIRECTORY": r'./data/sentiment_test/',  # Directory with the testing data
    "TWITTER_GLOVE": "./data/glove.twitter.27B/glove.twitter.27B.200d.txt",  # Glove twitter-specific dataset
    "GLOVE_DIR": 'glove.6B',  # Glove dataset that was tested before
    "MAX_SEQUENCE_LENGTH": 100,  # Max length of a tokenized twitter message
    "OUTPUT_CLASSES": 3,  # Number of classes for classification
    "MAX_NUM_WORDS": 2000 if DEBUG else 500000,  # Tokenizer max_num_words parameter
    "MAX_INDEX_CNT": 1000 if DEBUG else 1000000000,  # Max number of entries in the embeddings_index
    "EMBEDDING_DIM": 200,  # Dimensionality of the Embedding layer
    "VALIDATION_SPLIT": 0.2,  # Rate of splitting between train/test set
    "SYNONIMIZE_FRACTION": 0.2,  # Fraction of the batch that will be used in the synonimization augmenting function
    "SYNONIMIZE_WORDS_FRACTION": 0.1,  # Fraction of the message that will be be affected by the synonymization
    "SYNONIM_SIMILARITY_THR": 0.9,  # Threshold above which (in terms of similarity) synonim will be taken into
    # consideration
    "model_config": model_config,  # Dictionary that defines model configuration
    "BATCH_SIZE": 128,  # Size of the training batch
    "EPOCHS": 15,  # Number of epochs that will be used in the training
    "TEXT_PREPROCESSING": 1  # Will text processing be used before tokenization
}
