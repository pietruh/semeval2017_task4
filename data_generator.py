import keras
import numpy as np
import gensim.downloader as api
import random # for randomly sampling


class SynonymDataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""
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
        self.word_index_keys_as_list = list(word_index.keys())
        self.word_index_keys_as_arr = np.array(list(word_index.keys()))

        self.config = config
        # TODO: shuffle data on_epoch_end()

        # Load model straight from gensim to efficiently find most_similar words
        info = api.info()  # show info about available models/datasets
        self.model_emb = api.load("glove-twitter-25")  # download the model and return as object ready for use

        self.on_epoch_end()
        return


    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.train_data_x) / self.batch_size))


    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.train_data_x))
        return

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""

        # Initialization
        x = np.copy(self.train_data_x)
        #x = np.empty((self.batch_size, self.config["MAX_SEQUENCE_LENGTH"]))
        y = np.empty((self.batch_size, self.config["OUTPUT_CLASSES"]))

        # Here, draft some of the x-inputs and modify them
        # Parametrise the augmentation process

        # Determine which of the inputs will be modified by the synonimization process
        num_of_synon_arrays = np.floor(self.batch_size*self.config["SYNONIMIZE_FRACTION"])
        # TODO: finish data generation. Consider having augmentation function as a stand-alone fucntion not, neccesarilly member funciton.
        # TODO: Write "most_similar" for a dictionary of arrays -> this will find synonyms in the dataset. Also consider np/tf/keras built-in algorithm for that, like Nearest-neighbor for example.
        # TODO: This may be solved by loading pretrained model from keras.api that has it implemented
        # TODO: Set some threshold above which words are similar: experimentally test that

        #np.array
        in_batch_IDs = random.sample(indexes, num_of_synon_arrays)
        # Determine how many words will be synonymized in the sentence
        ## tweets_affected = self.train_data_x[in_batch_IDs]  # this should work
        # draft 2d array of random positions
        # for each tweet generate positions of words that will be impacted and store them in the array

        for one_tweet_id in in_batch_IDs:
            #find #SYNONIMIZE_WORDS non-zeros in the tweets
            one_tweet_seq = self.train_data_x[one_tweet_id]
            first_non_zero_pos = (one_tweet_seq == 0).argmin()
            num_words_to_change = int(self.config["SYNONIMIZE_WORDS_FRACTION"] * (len(one_tweet_seq) - first_non_zero_pos))
            positions = random.sample(range(first_non_zero_pos, len(one_tweet_seq)), num_words_to_change)
            #words_to_be_changed = self.word_index_keys_as_list[positions]
            tmp = one_tweet_seq[positions]
            words_to_be_changed = self.word_index_keys_as_arr[tmp - 1]  # words as text
            for word_ind in range(0, len(words_to_be_changed)):
                synonyms = self.model_emb.most_similar(words_to_be_changed[word_ind])[0]

                # Check if synonym is acceptable, if yes, then change original word id in the sequence for the synon_id
                if synonyms[0][1] > self.config["SYNONIM_SIMILARITY_THR"]:
                    synonym_id = self.word_index[synonyms[0][0]]
                    one_tweet_seq[positions[word_ind]] = synonym_id

            # assign modified sequence back to the x training data
            x[one_tweet_id] = one_tweet_seq
            # leave y the same
        return x, y

    def get_len_non_zero(self, one_tweet_text):
        return
