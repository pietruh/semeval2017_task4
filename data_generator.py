"""
Data generator class, that uses word synonimization during calling model.fit(). It dynamically creates augmented
examples on a batch that is used for training
"""
import keras
import numpy as np
import gensim.downloader as api
import random  # for randomly sampling


class SynonymDataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras by synonymizing words, and changing tweets a little. This may be used by the
    fit_generator() call to train the model.
    """

    def __init__(self, batch_size, train_data_x, train_data_y, embeddings_index, word_index, config):
        """Create object that will be passed to fit_generator, native keras function
        :param batch_size: Size of the batch that is being used in the fit_generator
        :type batch_size: int
        :param train_data_x: Training data, tweets as sequences of word_indexes, already padded, preprocessed etc. training ready
        :type train_data_x: 2D array of floats
        :param train_data_y: One-hot encoded ground-truth labels
        :type train_data_y: 2D array of floats
        :param embeddings_index: Dictionary in which there is word as a key, and value is n-dim embedding array
        :type embeddings_index: dict
        :param word_index: Look-up Table that returns word_id given the word
        :type word_index: dict
        :param config: Configuration object that defines some useful hyper-parameters
        :type config: dict
        """

        self.batch_size = batch_size
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.embeddings_index = embeddings_index
        self.word_index = word_index
        self.word_index_keys_as_list = list(word_index.keys())
        self.word_index_keys_as_arr = np.array(list(word_index.keys()))

        self.config = config
        # TODO: Consider shuffling data on_epoch_end()

        # Load model straight from gensim to efficiently find most_similar words
        info = api.info()  # show info about available models/datasets
        self.model_emb = api.load("glove-twitter-25")  # download the model and return as object ready for use

        self.on_epoch_end()
        print("self.__len__() = {}".format(self.__len__()))
        return

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.train_data_x) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.train_data_x))
        return

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""

        # Here, draft some of the x-inputs and modify them
        # Parametrise the augmentation process
        x, y = synonimize_data(self.train_data_x, self.train_data_y, indexes, self.batch_size,
                               self.word_index_keys_as_arr, self.word_index, self.model_emb, **self.config)

        return x, y

    def get_len_non_zero(self, one_tweet_text):
        return


def synonimize_data(train_data_x, train_data_y, indexes, batch_size, word_index_keys_as_arr, word_index, model_emb,
                    **kwargs):
    """
    Function that finds synonyms for some of the tweets in the given dataset.
    :param train_data_x:
    :param train_data_y:
    :param indexes:
    :param batch_size:
    :param word_index_keys_as_arr:
    :param word_index:
    :param model_emb:
    :param kwargs:
    :return:
    """
    SYNONIMIZE_FRACTION = kwargs.get("SYNONIMIZE_FRACTION", 0.2)
    SYNONIMIZE_WORDS_FRACTION = kwargs.get("SYNONIMIZE_WORDS_FRACTION", 0.2)
    SYNONIM_SIMILARITY_THR = kwargs.get("SYNONIM_SIMILARITY_THR", 0.9)

    x = np.copy(train_data_x[indexes])
    y = np.copy(train_data_y[indexes])

    # Determine which of the inputs will be modified by the synonimization process
    num_of_synon_arrays = np.floor(batch_size * SYNONIMIZE_FRACTION)
    in_batch_IDs = random.sample(list(indexes), int(num_of_synon_arrays))
    # Determine how many words will be synonymized in the sentence
    # draft 2d array of random positions
    # for each tweet generate positions of words that will be impacted and store them in the array
    tweet_cnt = 0
    for one_tweet_id in in_batch_IDs:
        # find how many words will be changed in the tweet
        one_tweet_seq = train_data_x[one_tweet_id]
        first_non_zero_pos = (one_tweet_seq == 0).argmin()
        num_words_to_change = int(SYNONIMIZE_WORDS_FRACTION * (len(one_tweet_seq) - first_non_zero_pos))
        positions = random.sample(range(first_non_zero_pos, len(one_tweet_seq)), num_words_to_change)
        tmp = one_tweet_seq[positions]
        words_to_be_changed = word_index_keys_as_arr[tmp - 1]  # words as text
        for word_ind in range(0, len(words_to_be_changed)):
            try:
                synonyms = model_emb.most_similar(words_to_be_changed[word_ind])[0]

                # Check if synonym is acceptable, if yes, then change original word id in the sequence for the synon_id
                if synonyms[1] > SYNONIM_SIMILARITY_THR:
                    synonym_id = word_index[synonyms[0]]
                    one_tweet_seq[positions[word_ind]] = synonym_id
            except:
                continue
        # assign modified sequence back to the x training data
        twt_pos_in_batch = one_tweet_id - indexes[0]
        x[twt_pos_in_batch] = one_tweet_seq

        # leave y the same
    return x, y
