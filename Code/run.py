import modules
import tensorflow as tf
import keras
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

# 1. LOAD DATA

# LOADING FEATURES AND IDS 
feat_arr_train, ids_train = modules.get_train_data()
feat_arr_test, ids_test, feat_arr_test_female, feat_arr_test_male, ids_test_female, ids_test_male = modules.get_test_data()
feat_arr_val, ids_val = modules.get_val_data()

print('----------------------------------------------------------------')
print()
print('Training Utterances: {}'.format(len(feat_arr_train)))
print('Testing Utterances: {}'.format(len(feat_arr_test)))
print('Validation Utterances: {}'.format(len(feat_arr_val)))
print()
print('----------------------------------------------------------------')

# LOADING LABELS
labels_train, labels_val, labels_test_female, labels_test_male = modules.get_labels(ids_train, ids_test, ids_val)

# CREATING A TF DATASET (for less memory usage and fast processing):

# batch_size = 32
# train_data = tf.data.Dataset.from_tensor_slices((feat_arr_train, labels_train)).shuffle(13170).batch(batch_size),
# #test_data = tf.data.Dataset.from_tensor_slices((feat_arr_test, y_val)).shuffle(1680).batch(batch_size),
# val_data = tf.data.Dataset.from_tensor_slices((feat_arr_val, labels_val)).shuffle(230).batch(batch_size)



# 2. BUILDING THE MODEL


# Choose any one of the following model and comment out the other two:

model = modules.cross_attn_model()
# model = modules.attn_model()
# model = modules.lstm_model()



# 3. TRAINING THE MODEL

# Shuffling the training dataset
feat_arr_train,labels_train = shuffle(feat_arr_train,labels_train, random_state=0)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_model_wts.h5', monitor='val_loss', mode='min', save_best_only=True,save_weights = True, verbose=1)

history = model.fit(np.array(feat_arr_train), labels_train.astype(float), 
                    validation_data = (np.array(feat_arr_val), labels_val.astype(float)), epochs=200, batch_size=32, callbacks=[es, mc])

model.load_weights('best_model_wts.h5')
modules.test(model, feat_arr_test_male, feat_arr_test_female, labels_test_male[:,1], labels_test_male[:,0], labels_test_female[:,1], labels_test_female[:,0])
