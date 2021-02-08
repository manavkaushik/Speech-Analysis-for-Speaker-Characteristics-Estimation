# Implementation of the proposed Cross-Attention model:

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Dense, Dropout, Permute, merge, Reshape, Flatten, AveragePooling1D
from keras.layers import LSTM, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from .soft_attention import attention
from .focal_loss import focal_loss
from .load_data import data_gen_train
from .load_data import data_gen_test
from .load_data import data_gen_val




def lstm_att_model():

    # DATA PIPELINING:

    ds_series_train = tf.data.Dataset.from_generator(
        data_gen_train, 
        output_types=(tf.float32, tf.float32),
        output_shapes=((1000, 83), (8)))
        
    ds_series_test = tf.data.Dataset.from_generator(
        data_gen_test, 
        output_types=(tf.float32, tf.float32),
        output_shapes=((1000, 83), (8)))
        
    ds_series_val = tf.data.Dataset.from_generator(
       data_gen_val, 
       output_types=(tf.float32, tf.float32),
       output_shapes=((1000, 83), (8)))



    print('Train Data: {}'.format(ds_series_train))
    print('Test Data: {}'.format(ds_series_test))
    print('Val Data: {}'.format(ds_series_val))

    ds_series_train_batch = ds_series_train.shuffle(125555).padded_batch(64)
    ds_series_test_batch = ds_series_test.shuffle(15000).padded_batch(64)
    ds_series_val_batch = ds_series_val.shuffle(11988).padded_batch(64)


    # MODEL ARCHITECTURE

    inputs = keras.Input(shape=(1000, 83,))

    lstm = LSTM(512, kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001), input_shape=(1000, 83), name = 'LSTM',
                                recurrent_regularizer=None, bias_regularizer=regularizers.l2(1e-4),
                                activity_regularizer=regularizers.l2(1e-5), kernel_constraint=None,
                                recurrent_constraint=None, bias_constraint=None, dropout=0.2, return_sequences=True)(inputs) 

    att, att_weight = attention()(lstm)
    reshaped = Reshape((512,))(K.sum(att,axis=1))
    #reshaped = Reshape((1512,))(tf.concat([K.sum(att,axis=1), K.sum(att,axis=2)], axis=1))

    drop1 = keras.layers.Dropout(0.2)(reshaped)
    dense = Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(0.001), 
                    activity_regularizer=tf.keras.regularizers.l2(0.01))(drop1)

    drop2 = keras.layers.Dropout(0.2)(dense)
    outputs = Dense(8, activation = 'softmax', kernel_regularizer=regularizers.l2(0.001), 
                    activity_regularizer=tf.keras.regularizers.l2(0.01))(drop2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=5, patience=4)
    mc = ModelCheckpoint('../../lstm_att_model_wts.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights = True, verbose=1)
    #tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq = 1)
        

    hist = model.fit(ds_series_train_batch, validation_data = ds_series_val_batch, epochs=1, callbacks=[es, mc])

    ############################ TESTING THE MODEL ######################################

    print()
    print()
    print('TESTING for LSTM ATTENTION with CROSSENTROPY LOSS.....................................')
    print()
    model.load_weights('../../lstm_att_model_wts.h5')
    model.evaluate(ds_series_test_batch)

    return model