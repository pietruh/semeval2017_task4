import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""
    def __init__(self, batch_size, train_data_x, train_data_y, embeddings_index, config):
        self.batch_size = batch_size
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.embeddings_index = embeddings_index
        self.config = config
        # TODO: shuffle data on_epoch_end()
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
        X, y = self.__data_generation(indexes, self.embeddings_index)
        return X, y


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.train_data_x))
        return

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""

        # Initialization
        x = np.empty((self.batch_size, self.config["MAX_SEQUENCE_LENGTH"]))
        y = np.empty((self.batch_size, self.config["OUTPUT_CLASSES"]))

        # Here, draft some of the x-inputs and modify them
        # Parametrise the augmentation process

        # Determine which of the inputs will be modified by the synonimization process
        num_of_synon_arrays = np.floor(self.batch_size*self.config["SYNONIMIZE_FRACTION"])
        #in_batch_IDs =

        return
