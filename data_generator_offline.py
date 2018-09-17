"""This script generates new data using technics of data augmentation 'before' fitting the model."""

from nlp_utilities import get_data, save_text_as_tweet
from data_generator import synonimize_data
from config import config, model_config, DEBUG
import numpy as np
import gensim.downloader as api
from nltk.corpus import wordnet as wn

import random

from nltk.tokenize import word_tokenize
import nltk


def synonimize_data_offline(text_data_x, text_data_y, indexes, batch_size, model_emb, **kwargs):


    SYNONIMIZE_FRACTION = kwargs.get("SYNONIMIZE_FRACTION", 0.2)
    SYNONIMIZE_WORDS_FRACTION = kwargs.get("SYNONIMIZE_WORDS_FRACTION", 0.2)
    SYNONIM_SIMILARITY_THR = kwargs.get("SYNONIM_SIMILARITY_THR", 0.9)



    # Determine which of the inputs will be modified by the synonimization process
    num_of_synon_arrays = np.floor(batch_size*SYNONIMIZE_FRACTION)
    in_batch_IDs = random.sample(list(indexes), int(num_of_synon_arrays))
    # return an array of tweets with its class
    y = np.copy(text_data_y[in_batch_IDs])
    #x = np.empty([batch_size])
    x = []
    tweet_cnt = 0
    for one_tweet_id in in_batch_IDs:
        #find how many words will be changed in the tweet
        one_tweet_seq = text_data_x[one_tweet_id]
        one_tweet_seq_tokenized = word_tokenize(one_tweet_seq)
        #first_non_zero_pos = (one_tweet_seq == 0).argmin()
        num_words_to_change = int(SYNONIMIZE_WORDS_FRACTION * len(one_tweet_seq_tokenized))
        positions = random.sample(range(0, len(one_tweet_seq_tokenized)), num_words_to_change)
        words_to_be_changed = np.array(list(one_tweet_seq_tokenized))[positions]  # words as text

        for word_ind in range(0, len(words_to_be_changed)):
            try:
                synonyms = model_emb.most_similar(words_to_be_changed[word_ind])[0]

                # Check if synonym is acceptable, if yes, then change original word id in the sequence for the synon_id
                if synonyms[1] > SYNONIM_SIMILARITY_THR:
                    one_tweet_seq_tokenized[positions[word_ind]] = synonyms[0]
            except:
                # If the word does not exist in the dictionary, just omit it
                continue

        # assign modified sequence back to the x training data
        changed_tweet = ' '.join(word for word in one_tweet_seq_tokenized)
        x.append(changed_tweet)
        #x[tweet_cnt] = changed_tweet
        tweet_cnt += 1


        # leave y the same
    return x, y


def hypernymize_data_offline(text_data_x, text_data_y, indexes, batch_size, **kwargs):
    print("Running hypernymize_data_offline")

    HYPERNIMIZE_FRACTION = kwargs.get("HYPERNIMIZE_FRACTION", 0.2)
    HYPERNIMIZE_WORDS_FRACTION = kwargs.get("HYPERNIMIZE_WORDS_FRACTION", 0.2)

    # Determine which of the inputs will be modified by the synonimization process
    num_of_synon_arrays = np.floor(batch_size * HYPERNIMIZE_FRACTION)
    in_batch_IDs = random.sample(list(indexes), int(num_of_synon_arrays))
    # return an array of tweets with its class
    y = np.copy(text_data_y[in_batch_IDs])
    x = []
    tweet_cnt = 0
    for one_tweet_id in in_batch_IDs:
        # find how many words will be changed in the tweet
        one_tweet_seq = text_data_x[one_tweet_id]
        one_tweet_seq_tokenized = word_tokenize(one_tweet_seq)

        num_words_to_change = int(HYPERNIMIZE_WORDS_FRACTION * len(one_tweet_seq_tokenized))
        positions = random.sample(range(0, len(one_tweet_seq_tokenized)), num_words_to_change)
        words_to_be_changed = np.array(list(one_tweet_seq_tokenized))[positions]  # words as text

        for word_ind in range(0, len(words_to_be_changed)):
            try:
                hypernym = get_hypernym(words_to_be_changed[word_ind])
                # Check if synonym is acceptable, if yes, then change original word id in the sequence for the synon_id
                if hypernym:
                    one_tweet_seq_tokenized[positions[word_ind]] = hypernym[0]
            except:
                # If the word does not exist in the dictionary, just omit it
                continue

        # assign modified sequence back to the x training data
        changed_tweet = ' '.join(word for word in one_tweet_seq_tokenized)
        x.append(changed_tweet)
        tweet_cnt += 1

        # leave y the same
    return x, y

def get_words_to_be_changed(text_data_x, one_tweet_id, HYPERNIMIZE_WORDS_FRACTION):
    one_tweet_seq = text_data_x[one_tweet_id]
    one_tweet_seq_tokenized = word_tokenize(one_tweet_seq)

    num_words_to_change = int(HYPERNIMIZE_WORDS_FRACTION * len(one_tweet_seq_tokenized))
    positions = random.sample(range(0, len(one_tweet_seq_tokenized)), num_words_to_change)
    words_to_be_changed = np.array(list(one_tweet_seq_tokenized))[positions]  # words as text
    return words_to_be_changed

def get_wn_tag(tag):
    tag_dict = {'NN': 'n', 'JJ': 'a',
                'VB': 'v', 'RB': 'r'}
    try:
        return tag_dict[tag[0][1]]
    except:
        return None#

def get_hypernym(word):
    text = word_tokenize(word)
    pos_tag = nltk.pos_tag(text)
    pos_tag_wn = get_wn_tag(pos_tag)

    try:
        hypernym = [lemma.name() for synset in wn.synset(word+'.'+ pos_tag_wn +'.01').hypernyms() for lemma in synset.lemmas()][0]
    except:
        hypernym = None
    return hypernym
# region main
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # ss data (tokenizer will be loaded inside this function)
    # this will load text in the
    print("Loading and preprocessing data")
    synon_or_hyper = 0     # 0 for synonimization, 1 for hypernimization

    config["SYNONIMIZE_FRACTION"] = 0.1
    config["SYNONIMIZE_WORDS_FRACTION"] = 0.3
    config["SYNONIM_SIMILARITY_THR"] = 0.9

    config["HYPERNIMIZE_FRACTION"] = 0.1
    config["HYPERNIMIZE_WORDS_FRACTION"] = 0.1

    train_directory = r'./data/sentiment_train/'

    test_data, test_labels, test_word_index, test_tokenizer, test_texts = get_data(train_directory, config,
                                                                                   tokenizer=None,
                                                                                   mode="test")
    sequences = test_tokenizer.texts_to_sequences(test_texts)
    test_word_index_keys_as_arr = np.array(list(test_word_index.keys()))

    # Load model straight from gensim to efficiently find most_similar words
    info = api.info()  # show info about available models/datasets
    model_emb = api.load("glove-twitter-25")  # download the model and return as object ready for use
    indexes = np.arange(len(test_data))

    if synon_or_hyper:
        augmented_filename = train_directory + "augmented_training_SYNONYM_SF{0}_SWF{1}_SN_THR{2}.txt".format(
            config["SYNONIMIZE_FRACTION"],
            config["SYNONIMIZE_WORDS_FRACTION"],
            config["SYNONIM_SIMILARITY_THR"]).replace(".", "p")
        x, y = synonimize_data_offline(test_texts, test_labels, indexes, len(test_texts), model_emb, **config)
    else:
        augmented_filename = train_directory + "augmented_training_HYPERNYM_HF{0}_HWF{1}.txt".format(
            config["HYPERNIMIZE_FRACTION"],
            config["HYPERNIMIZE_WORDS_FRACTION"]).replace(".", "p")
        x, y = hypernymize_data_offline(test_texts, test_labels, indexes, len(test_texts), **config)

    save_text_as_tweet(x, y, augmented_filename)


    # Synonymize data
   # x, y = synonimize_data(test_data, test_labels, indexes, config["BATCH_SIZE"], test_word_index_keys_as_arr, test_word_index, model_emb,
    #                       **config)
    # TODO: get back to the text!


    # save results to the augmented text file


# ----------------------------------------------------------------------------------------------------------------------
# endregion main
