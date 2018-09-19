"""This script generates new data using techniques of data augmentation 'before' fitting the model."""

from config import config
import numpy as np
import gensim.downloader as api
from nltk.corpus import wordnet as wn
import sys
import random
from nltk.tokenize import word_tokenize
import nltk
# My custom objects and functions
from logger_to_file import Logger
from nlp_utilities import get_data, save_text_as_tweet


# region augmentation_utilities
# ----------------------------------------------------------------------------------------------------------------------

def augment_data_offline(text_data_x, text_data_y, indexes, batch_size, type="synon", **kwargs):
    if type == "syno":
        FRACTION = kwargs.get("SYNONIMIZE_FRACTION", 0.2)
        WORDS_FRACTION = kwargs.get("SYNONIMIZE_WORDS_FRACTION", 0.2)
        SYNONIM_SIMILARITY_THR = kwargs.get("SYNONIM_SIMILARITY_THR", 0.9)

        # Load model straight from gensim to efficiently find most_similar words
        info = api.info()  # show info about available models/datasets
        model_emb = api.load("glove-twitter-25")  # download the model and return as object ready for use
    elif type == "hyper":
        FRACTION = kwargs.get("HYPERNIMIZE_FRACTION", 0.2)
        WORDS_FRACTION = kwargs.get("HYPERNIMIZE_WORDS_FRACTION", 0.2)

    # Determine which of the inputs will be modified by the synonimization process
    num_of_synon_arrays = np.floor(batch_size * FRACTION)
    in_batch_IDs = random.sample(list(indexes), int(num_of_synon_arrays))

    # Return an array of tweets with its class
    y = np.copy(text_data_y[in_batch_IDs])
    x = []
    tweet_cnt = 0
    # Loop over randomly chosen ID to modify them
    for one_tweet_id in in_batch_IDs:
        # Find how many words will be changed in the tweet
        one_tweet_seq = text_data_x[one_tweet_id]
        one_tweet_seq_tokenized = word_tokenize(one_tweet_seq)

        num_words_to_change = int(WORDS_FRACTION * len(one_tweet_seq_tokenized))
        positions = random.sample(range(0, len(one_tweet_seq_tokenized)), num_words_to_change)
        words_to_be_changed = np.array(list(one_tweet_seq_tokenized))[positions]  # words as text

        for word_ind in range(0, len(words_to_be_changed)):
            # Iterate over the wordsto be changed to modify them in a single twitter message
            #if words_to_be_changed[word_ind] not in("<hashtag>")
            if type == "syno":
                # Find synonym of the word using provided model
                try:
                    synonyms = model_emb.most_similar(words_to_be_changed[word_ind])[0]
                    # Check if synonym is acceptable, if yes, then change original word id in the sequence for it
                    if synonyms[1] > SYNONIM_SIMILARITY_THR:
                        one_tweet_seq_tokenized[positions[word_ind]] = synonyms[0]
                except:
                    # If the word does not exist in the dictionary, or some dict error was raised, just omit it
                    continue

            elif type == "hyper":
                # Find hypernym (word that describes more general concept) of a word, using dictionary
                try:
                    hypernym = get_hypernym(words_to_be_changed[word_ind])
                    # Check if hypernym is acceptable, if yes, then change original word id in the sequence for it
                    if hypernym:
                        one_tweet_seq_tokenized[positions[word_ind]] = hypernym
                except:
                    # If the word does not exist in the dictionary, or some dict error was raised, just omit it
                    continue

        # Assign modified sequence back to the x training data
        changed_tweet = ' '.join(word for word in one_tweet_seq_tokenized)
        x.append(changed_tweet)
        print("Original:   {}\n".format(one_tweet_seq))
        print("Changed:    {}\n\n".format(changed_tweet))
        tweet_cnt += 1
        # leave y the same
    return x, y


def get_wn_tag(tag):
    tag_dict = {'NN': 'n', 'JJ': 'a',
                'VB': 'v', 'RB': 'r'}
    try:
        return tag_dict[tag[0][1]]
    except:
        return None  #


def get_hypernym(word):
    text = word_tokenize(word)
    pos_tag = nltk.pos_tag(text)
    pos_tag_wn = get_wn_tag(pos_tag)

    try:
        hypernym = [lemma.name() for synset in wn.synset(word + '.' + pos_tag_wn + '.01').hypernyms() for lemma in
                    synset.lemmas()][0]
    except:
        hypernym = None
    return hypernym


# ----------------------------------------------------------------------------------------------------------------------
# endregion augmentation_utilities


# region main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define train directory in which new data will be added
    train_directory = r'./data/sentiment_train/'
    sys.stdout = Logger(train_directory + "log_augmenting")

    # ss data (tokenizer will be loaded inside this function)
    # this will load text in the
    print("Loading and preprocessing data")

    # Set up augmentation technique, and its parameters
    augment_type = "hyper"  # "hyper" for hypernimization, "syno" for synonimization
    config["SYNONIMIZE_FRACTION"] = 0.05
    config["SYNONIMIZE_WORDS_FRACTION"] = 0.3
    config["SYNONIM_SIMILARITY_THR"] = 0.85

    config["HYPERNIMIZE_FRACTION"] = 0.05
    config["HYPERNIMIZE_WORDS_FRACTION"] = 0.35

    # Get data from train directory to be used as a augmentation base
    test_data, test_labels, test_word_index, test_tokenizer, test_texts = get_data(train_directory, config,
                                                                                   tokenizer=None,
                                                                                   mode="test")
    sequences = test_tokenizer.texts_to_sequences(test_texts)
    test_word_index_keys_as_arr = np.array(list(test_word_index.keys()))
    indexes = np.arange(len(test_data))

    # depending on the technique used, augment_data_offline will provide augmented training dataset
    if augment_type == "syno":
        augmented_filename = train_directory + "augmented_training_PREPROCESSED_SYNONYM_SF{0}_SWF{1}_SN_THR{2}.txt" \
            .format(
            config["SYNONIMIZE_FRACTION"],
            config["SYNONIMIZE_WORDS_FRACTION"],
            config["SYNONIM_SIMILARITY_THR"]).replace(".", "p") + ".txt"
        x, y = augment_data_offline(test_texts, test_labels, indexes, len(test_texts), type="syno", **config)
    elif augment_type == "hyper":
        augmented_filename = train_directory + "augmented_training_PREPROCESSED_HYPERNYM_HF{0}_HWF{1}".format(
            config["HYPERNIMIZE_FRACTION"],
            config["HYPERNIMIZE_WORDS_FRACTION"]).replace(".", "p") + ".txt"
        x, y = augment_data_offline(test_texts, test_labels, indexes, len(test_texts), type="hyper", **config)

    # Save results to the augmented text file
    save_text_as_tweet(x, y, augmented_filename)
    print("Data augmentation finished successfully.")
# ----------------------------------------------------------------------------------------------------------------------
# endregion main
