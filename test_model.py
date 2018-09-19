"""Script for evaluating trained model on any dataset that has formatting complaint with SEMEVAL2017 TASK4-A"""

from keras.models import load_model
from nlp_utilities import gpu_configuration_initialization, get_data, save_predictions, macro_averaged_recall_tf_onehot, macro_averaged_recall_tf_soft, save_text_as_tweet
from config import config, model_config, DEBUG  # no-brainer version of setting up configuration for testing
from keras.utils.generic_utils import get_custom_objects

if __name__ == "__main__":
    eval_or_predict = 2  # 0 for evaluation only, 1 for saving predictions ot a file, 2 for both
    gpu_configuration_initialization()
    custom_objects = {'macro_averaged_recall_tf_soft': macro_averaged_recall_tf_soft,
                      'macro_averaged_recall_tf_onehot': macro_averaged_recall_tf_onehot
                      }

    get_custom_objects().update(custom_objects)

    NAME_OF_THE_TRAIN_SESSION = "7_testing_augmented_data_full_model+gaussian+macro_loss_+full_data"
    PATH_TO_THE_LEARNING_SESSION = "./learning_sessions/" + NAME_OF_THE_TRAIN_SESSION + "/"

    PATH_TO_THE_MODEL = PATH_TO_THE_LEARNING_SESSION + "rnn_model_3.h5"  # path to the model
    TEST_DIRECTORY = r'./data/sentiment_test/'  # directory that contains test set
    config["TOKENIZER_PATH"] = PATH_TO_THE_LEARNING_SESSION + "tokenizer.pickle"

    #path_to_the_best_weights = PATH_TO_THE_LEARNING_SESSION + "weights_ckpt.hdf5"

    # load model
    print("Loading the model")
    model = load_model(PATH_TO_THE_MODEL)
    #print("Loading the updated weights")
    #model.load_weights(path_to_the_best_weights)

    # load & preprocess data (tokenizer will be loaded inside this function)
    print("Loading and preprocessing data")
    test_data, test_labels, test_word_index, test_tokenizer, test_texts = get_data(TEST_DIRECTORY, config, tokenizer=None,
                                                                       mode="test")

    if eval_or_predict in (0, 2):
        # evaluate the model
        print("Evaluating the model")
        loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])

        print("Evaluated metrics\n values: {} \n metrics: {}".format(loss_metrics_eval, model.metrics_names))
    if eval_or_predict in (1, 2):
        print("Predicting outputs to file")
        test_predictions = model.predict(test_data)
        # save model output from the softmax layer
        prediction_filename = PATH_TO_THE_LEARNING_SESSION + "predictions.txt"
        save_predictions(prediction_filename, test_predictions)
        print("Predictions from softmax saved to file {}".format(prediction_filename))
        # save output class (that has the biggest probability according to the model) with the text of tweets
        prediction_as_original = PATH_TO_THE_LEARNING_SESSION + "predictions_as_tweets.txt"
        save_text_as_tweet(test_texts, test_predictions, prediction_as_original)
        print("Predictions with text saved to file {}".format(prediction_as_original))

    print("Testing finished successfully")
