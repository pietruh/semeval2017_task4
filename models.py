"""
Definition of a models and model utilities are inside this file.
"""

from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation, Flatten, GaussianNoise
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import BatchNormalization



# region models
# ----------------------------------------------------------------------------------------------------------------------
def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    """
    Short function that abstracts recurrent layers generation and returns ready-to-go layer.
    :param unit: Name of the recurrent neuron unit that will be used in the layer.
    :param cells: Number of cells (units) for the layer.
    :param bi: Bidirectional or one directional connections between cells.
    :param return_sequences: Access the sequence of hidden state outputs.
    :param dropout_U: Fraction of the input units to drop for recurrent connections.
    :param consume_less: Optimization setting.
    :param l2_reg: Weights regularizer.
    :returns: Recurrent Layer
    :rtype: keras.layers
    """
    rnn = unit(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn


def get_RNN_model_w_layer(embedding_layer, macro_averaged_recall_tf_onehot, macro_averaged_recall_tf_soft):
    """
    First approach to model with Recurrent Units - LSTMs. This may be very dirty and hacky.
    This function provides declaration of the model that will be used to fit on data.
    Compiled, ready-to-go model will be returned.
    :param embedding_layer: Already pre-trained and pre-loaded Embedding layer that will be used to map the input words
                            IDs to the vector space.
    :type embedding_layer: keras.layers.Embedding
    :returns: Compiled model, ready for fitting to data
    :rtype: keras.models.Sequential
    """
    # TODO: Pass scoring functions in a different way.
    # TODO: Pass config dict with parsing here. Leaving here just for fast prototyping
    classes = 3  # Output classes

    dropout_embeddings = 0.3  # Dropout rate after the embedding layer
    dropout_U = 0.3  # Dropout rate in the recurrent layer
    dropout_rnn = 0.3  # Dropout rate after the recurrent layer

    loss_l2 = 0.0001  # L2 regularization penalty to the loss function to discourage large weights(weight decay)
    clipnorm = 5.  # Clip the norm of the gradients as an extra safety measure against exploding gradients
    lr = 0.001  # Starting learning rate for Adam optimizer
    gaussian_noise = 0.2  # Random data augmentation technique, making this model more robust to overfitting

    # Create sequential model
    model = Sequential()

    # Perturb inputs with gaussian noise, to make it more robust
    # Define embedding input layer
    model.add(embedding_layer)
    model.add(BatchNormalization())
    model.add(GaussianNoise(gaussian_noise))

    # Add Dropout for regularization after embedding layer
    model.add(Dropout(dropout_embeddings))

    # Define bidirectional LSTM layer no. 1
    rnn_layer_orig = get_RNN(cells=150, bi=True, dropout_U=dropout_U)
    model.add(rnn_layer_orig)
    # Add dropout for regularization after LSTM layer no. 1
    model.add(Dropout(dropout_rnn))
    model.add(BatchNormalization())

    # Define bidirectional LSTM layer no. 2

    rnn_layer_orig_2 = get_RNN(cells=150, bi=True, dropout_U=dropout_U)
    model.add(rnn_layer_orig_2)
    # Add dropout for regularization after LSTM layer no. 2
    model.add(Dropout(dropout_rnn))

    # Add Dense layer, fully connected with softmax activation
    model.add(Flatten())
    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    # Compile the model using proided metrics and loss
    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss=macro_averaged_recall_tf_soft,
                  metrics=[macro_averaged_recall_tf_onehot, macro_averaged_recall_tf_soft, 'categorical_crossentropy',
                           'acc'])

    # Return compiled model
    return model


# ----------------------------------------------------------------------------------------------------------------------
# endregion models
