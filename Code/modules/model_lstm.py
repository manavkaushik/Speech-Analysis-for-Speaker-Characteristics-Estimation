# Implemenation of our Base LSTM Model:

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Dropout, Permute, merge, Reshape, Flatten, AveragePooling1D
from keras.layers import LSTM, Bidirectional
from .soft_attention import attention
from .loss_func import custom_loss_function

def lstm_model():

    # MODEL ARCHITECTURE

    inputs = keras.Input(shape=(800, 84,))

    # You may want to fine tune the parameters of this lstm layer for best performance
    lstm = LSTM(128, input_shape = (800, 84), activation='tanh', recurrent_activation='sigmoid',
        use_bias=True, kernel_initializer='glorot_uniform',
        recurrent_initializer='orthogonal',
        bias_initializer='zeros', unit_forget_bias=True,
        kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None,
        activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
        bias_constraint=None, dropout=0.2, recurrent_dropout=0.2,
        return_sequences=True, return_state=False, go_backwards=False, stateful=False,
        time_major=False, unroll=False)(inputs) 

    flatten = Flatten()(lstm)

    drop = Dropout(0.2)(flatten)
    dense = Dense(64, activation = 'relu')(flatten)

    outputs = Dense(2, activation = 'relu')(dense)

    model = keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    model.compile(loss= custom_loss_function, optimizer='adam', metrics=['mse','mae'])

    return model