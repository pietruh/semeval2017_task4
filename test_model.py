"""Script for evaluating trained model on any dataset that has formatting complaint with SEMEVAL2017 TASK4-A"""

from keras.models import load_model
from nlp_utilities import get_data, save_predictions, macro_averaged_recall_tf, macro_averaged_recall_tf_soft
from config import config, model_config, DEBUG  # no-brainer version of setting up configuration for testing
from keras.utils.generic_utils import get_custom_objects

if __name__ == "__main__":
    eval_or_predict = 2  # 0 for evaluation only, 1 for saving predictions ot a file, 2 for both

    custom_objects = {'macro_averaged_recall_tf': macro_averaged_recall_tf,
                      'macro_averaged_recall_tf_soft': macro_averaged_recall_tf_soft
                      }

    get_custom_objects().update(custom_objects)

    NAME_OF_THE_TRAIN_SESSION = "1_test_session"
    PATH_TO_THE_LEARNING_SESSION = "./learning_sessions/" + NAME_OF_THE_TRAIN_SESSION + "/"

    PATH_TO_THE_MODEL = PATH_TO_THE_LEARNING_SESSION + "rnn_model.h5"  # path to the model
    TEST_DIRECTORY = r'./data/sentiment_test/'  # directory that contains test set
    config["TOKENIZER_PATH"] = "./tokenizer.pickle"

    # load model
    print("Loading the model")
    model = load_model(PATH_TO_THE_MODEL)

    # load & preprocess data (tokenizer will be loaded inside this function)
    print("Loading and preprocessing data")
    test_data, test_labels, test_word_index, test_tokenizer = get_data(TEST_DIRECTORY, config, tokenizer=None,
                                                                       mode="test")

    if eval_or_predict in (0, 2):
        # evaluate the model
        print("Evaluating the model")
        loss_metrics_eval = model.evaluate(x=test_data, y=test_labels, batch_size=config["BATCH_SIZE"])

        print("Evaluated metrics\n values: {} \n metrics: {}".format(loss_metrics_eval, model.metrics_names))
    if eval_or_predict in (1, 2):
        print("Predicting outputs to file")
        test_predictions = model.predict(test_data)
        prediction_filename = TEST_DIRECTORY + "predictions.txt"
        save_predictions(prediction_filename, test_predictions)
        print("Predictions saved to file {}".format(prediction_filename))

    print("Testing finished successfully")
