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

config = {"GLOVE_DIR": 'glove.6B',
          "MAX_SEQUENCE_LENGTH": 100,
          "OUTPUT_CLASSES": 3,
          "MAX_NUM_WORDS": 2000 if DEBUG else 500000,
          "MAX_INDEX_CNT": 1000 if DEBUG else 1000000000,
          "EMBEDDING_DIM": 200,
          "VALIDATION_SPLIT": 0.2,
          "SYNONIMIZE_FRACTION": 0.2,
          "SYNONIMIZE_WORDS": 4,
          "SYNONIMIZE_WORDS_FRACTION": 0.1,
          "SYNONIM_SIMILARITY_THR": 0.9,
          "model_config": model_config,
          "BATCH_SIZE": 128,
          "EPOCHS": 15,
          "TEXT_PREPROCESSING": 1
          }