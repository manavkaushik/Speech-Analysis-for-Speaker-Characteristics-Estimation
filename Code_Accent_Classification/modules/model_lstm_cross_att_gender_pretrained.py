# Implementation of the proposed Cross-Attention model:

import os
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
from .load_data_gender import data_gen_train_1
from .load_data_gender import data_gen_train_0
from .load_data_gender import data_gen_test_1
from .load_data_gender import data_gen_test_0
from .load_data_gender import data_gen_val_1
from .load_data_gender import data_gen_val_0



def lstm_cross_att_gender_pretrain_model():

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

    ds_series_train_batch = ds_series_train.shuffle(125555).padded_batch(64)
    ds_series_test_batch = ds_series_test.shuffle(15000).padded_batch(64)
    ds_series_val_batch = ds_series_val.shuffle(11988).padded_batch(64)



    ds_series_train_0 = tf.data.Dataset.from_generator(
        data_gen_train_0, 
        output_types=((tf.float64, tf.float64)),
        output_shapes=(((1000, 83), (8))) )
        
    ds_series_train_1 = tf.data.Dataset.from_generator(
        data_gen_train_1, 
        output_types=((tf.float64, tf.float64)),
        output_shapes=(((1000, 83), (8))) )
        
        
        
    ds_series_test_0 = tf.data.Dataset.from_generator(
        data_gen_test_0, 
        output_types=((tf.float64, tf.float64)),
        output_shapes=(((1000, 83), (8))) )
        
    ds_series_test_1 = tf.data.Dataset.from_generator(
        data_gen_test_1, 
        output_types=((tf.float64, tf.float64)),
        output_shapes=(((1000, 83), (8))) )
        

        
    ds_series_val_0 = tf.data.Dataset.from_generator(
        data_gen_val_0, 
        output_types=((tf.float64, tf.float64)),
        output_shapes=(((1000, 83), (8))) )

    ds_series_val_1 = tf.data.Dataset.from_generator(
        data_gen_val_1, 
        output_types=((tf.float64, tf.float64)),
        output_shapes=(((1000, 83), (8))) )
    

    ds_series_train_batch_0 = ds_series_train_0.shuffle(70000).padded_batch(32)
    ds_series_test_batch_0 = ds_series_test_0.shuffle(8000).padded_batch(32)
    ds_series_val_batch_0 = ds_series_val_0.shuffle(7000).padded_batch(32)

    ds_series_train_batch_1 = ds_series_train_1.shuffle(70000).padded_batch(32)
    ds_series_test_batch_1 = ds_series_test_1.shuffle(8000).padded_batch(32)
    ds_series_val_batch_1 = ds_series_val_1.shuffle(7000).padded_batch(32)


    # MODEL ARCHITECTURE

    ######################################### MODEL PRE TRAINING ###############################################

    inputs = keras.Input(shape=(1000, 83,))

    lstm = LSTM(256, kernel_regularizer=regularizers.l1_l2(l1=0.00001, l2=0.00001), input_shape=(1000, 83), name = 'lstm',
                                recurrent_regularizer=None, bias_regularizer=regularizers.l2(1e-4),
                                activity_regularizer=regularizers.l2(1e-5), kernel_constraint=None,
                                recurrent_constraint=None, bias_constraint=None, dropout=0.2, return_sequences=True)(inputs) 

    att, att_weight = attention(name = 'attn')(lstm)
    #reshaped = Reshape((512,))(K.sum(att,axis=1))
    reshaped = Reshape((1256,))(tf.concat([K.sum(att,axis=1), K.sum(att,axis=2)], axis=1))

    drop1 = keras.layers.Dropout(0.2)(reshaped)
    dense = Dense(128, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.001), name = 'dense_1', activity_regularizer=tf.keras.regularizers.l2(0.01))(drop1)

    drop2 = keras.layers.Dropout(0.2)(dense)
    output_1 = Dense(8, activation = 'softmax', name='accent', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=tf.keras.regularizers.l2(0.01))(drop2)
    #output_2 = Dense(1, activation='relu', name = 'gender', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=tf.keras.regularizers.l2(0.01))(drop2)

    model = keras.Model(inputs=inputs, outputs=[output_1])
    model.summary()
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=5, patience=20)
    mc = ModelCheckpoint('../../lstm_cross_att_pretrain_model_wts.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights = True, verbose=1)
    #tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq = 1)
        
    model.load_weights('Accent_Data/gender_pre_trained_model.h5', by_name = True)
    hist = model.fit(ds_series_train_batch, validation_data = ds_series_val_batch, epochs=100, callbacks=[es, mc])

    ############################ TESTING THE MODEL ######################################

    print()
    print()
    print('TESTING for LSTM ATTENTION PRE-TRAINING for GENDER SPECIFIC MODEL.....................................')
    print()
    model.load_weights('../../lstm_cross_att_pretrain_model_wts.h5')
    model.evaluate(ds_series_test_batch)


    ############################################ MODEL FINE TUNING for Gender_1 ####################################################

    os.chdir('../../')
    mc_1 = ModelCheckpoint('../../lstm_cross_att_pretrain_model_wts_1.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights = True, verbose=1)
    model.load_weights('lstm_cross_att_pretrain_model_wts.h5', by_name=True)  
    hist_1 = model.fit(ds_series_train_batch_1, validation_data = ds_series_val_batch_1, epochs=100, callbacks=[es, mc_1])

    print()
    print()
    print('TESTING for Gender: 1 .....................................')
    print()
    print()
    model.load_weights('../../lstm_cross_att_pretrain_model_wts_1.h5')
    model.evaluate(ds_series_test_batch_1)


    ############################################ MODEL FINE TUNING for Gender_0 ####################################################

    os.chdir('../../')
    mc_0 = ModelCheckpoint('../../lstm_cross_att_pretrain_model_wts_0.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights = True, verbose=1)
    model.load_weights('lstm_cross_att_pretrain_model_wts.h5', by_name=True)  
    hist_0 = model.fit(ds_series_train_batch_0, validation_data = ds_series_val_batch_0, epochs=100, callbacks=[es, mc_0])

    print()
    print()
    print('TESTING for Gender: 0 .....................................')
    model.load_weights('../../lstm_cross_att_pretrain_model_wts_0.h5')
    model.evaluate(ds_series_test_batch_0)




    return model
