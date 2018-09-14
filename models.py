from keras.layers import Embedding, Bidirectional, LSTM, Dropout, MaxoutDense, Dense, Activation, Flatten

from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences,
               consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn

def get_RNN_model(embedding):
    """First approach to model with Recurrent Units - LSTMs. This may be very dirty and hacky"""
    classes = 3
    max_length = 50
    masking = True
    return_sequences = True
    consume_less = 'cpu'
    dropout_U = 0.3
    model = Sequential()
    dropout_rnn = 0.3
    dropout_final = 0.5
    loss_l2 = 0.0001
    clipnorm = 1.
    lr = 0.001
    # define embedding input layer

    # TODO: This have to be worked-around as shown in https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    embedding = Embedding(
        input_dim=embeddings.shape[0],
        output_dim=embeddings.shape[1],
        input_length=max_length if max_length > 0 else None,
        trainable=False,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )
    model.add(embedding)

    rnn_layer_orig = get_RNN()
    model.add(rnn_layer_orig)

    # define bidirectional LSTM layer no. 1

    # Add dropout for regularization after LSTM layer no. 1
    model.add(Dropout(dropout_rnn))

    # define bidirectional LSTM layer no. 2
    rnn_layer_orig_2 = get_RNN()

    # model.add(rnn_layer_orig_2)
    # # define bidirectional LSTM layer no. 2
    # rnn_2 = Bidirectional(LSTM(64, return_sequences=return_sequences,
    #                            consume_less=consume_less, dropout_U=dropout_U,
    #                            W_regularizer=l2(0.)))
    # model.add(rnn_2)
    # Add dropout for regularization after LSTM layer no. 2
    model.add(Dropout(dropout_rnn))

    # model.add(MaxoutDense(100, input_dim=(None, 50), W_constraint=maxnorm(2)))
    #
    # #TODO(MP): Until this it should work
    # model.add(Dropout(dropout_final))


    # define bidirectional LSTM layer no. 1

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy')
    return model



def get_RNN_model_w_layer(embedding_layer, macro_averaging_recall, macro_averaged_recall_tf_soft):
    """First approach to model with Recurrent Units - LSTMs. This may be very dirty and hacky"""
    #TODO: Config dict with parsing here
    classes = 3
    max_length = 50
    masking = True
    return_sequences = True
    consume_less = 'cpu'
    dropout_U = 0.3
    model = Sequential()
    dropout_rnn = 0.3
    dropout_final = 0.5
    loss_l2 = 0.0001
    clipnorm = 1.
    lr = 0.001
    # define embedding input layer

    model.add(embedding_layer)

    rnn_layer_orig = get_RNN()
    model.add(rnn_layer_orig)

    # define bidirectional LSTM layer no. 1

    # Add dropout for regularization after LSTM layer no. 1
    model.add(Dropout(dropout_rnn))

    # define bidirectional LSTM layer no. 2
    rnn_layer_orig_2 = get_RNN()

    model.add(rnn_layer_orig_2)
    # # define bidirectional LSTM layer no. 2
    # rnn_2 = Bidirectional(LSTM(64, return_sequences=return_sequences,
    #                            consume_less=consume_less, dropout_U=dropout_U,
    #                            W_regularizer=l2(0.)))
    # model.add(rnn_2)
    # Add dropout for regularization after LSTM layer no. 2
    model.add(Dropout(dropout_rnn))

    # model.add(MaxoutDense(100, input_dim=(None, 50), W_constraint=maxnorm(2)))
    #
    # #TODO(MP): Until this it should work
    # model.add(Dropout(dropout_final))


    # define bidirectional LSTM layer no. 1
    model.add(Flatten())
    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                  loss='categorical_crossentropy', metrics=[macro_averaging_recall, macro_averaged_recall_tf_soft, 'categorical_crossentropy', 'acc'])  # use this:macro_averaging_recall
    return model