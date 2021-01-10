import numpy as np
import os
from kaldiio import ReadHelper
from .spec_aug import spec_augment 
from keras.preprocessing.sequence import pad_sequences


# FOR TRAINING DATA

def get_train_data():

    feat_arr_train = []
    ids_train = []

    
    root_dir_train = 'TIMIT_Data/speed_perturbation_80fbanks3pitchs/dump/trainNet_sp/deltafalse/' # Root directory for training set
    os.chdir(root_dir_train)

    for i in range(1,21):
        with ReadHelper('scp:' + 'feats' + str(i) + '.scp') as reader:
            for key, numpy_array in reader:
                ids_train.append(key)
                numpy_array = spec_augment(numpy_array)
                
                if numpy_array.shape[0] < 800:
                    numpy_array = pad_sequences(numpy_array.T, maxlen=800, padding='post')
                    numpy_array = numpy_array.T
                
                elif numpy_array.shape[0] > 800:
                    numpy_array = numpy_array[:800]
                    

                # Incorporating gender information as a binary feature:

                if key[0] == 'F': # i.e. for female samples
                    #female_feat_arr_train.append(numpy_array)
                    #feat_arr_train.append(numpy_array)
                    feat_arr_train.append(np.concatenate((numpy_array, np.array([1]*800).reshape(800,1)), axis=1))

                elif key[0] == 'M': # i.e. for male samples
                    #male_feat_arr_train.append(numpy_array)
                    #feat_arr_train.append(numpy_array)
                    feat_arr_train.append(np.concatenate((numpy_array, np.array([0]*800).reshape(800,1)), axis=1))
                else:
                    print('ERROR! ' + str(key))

    # Coming back to main directory
    os.chdir('../../../../../')

    return feat_arr_train, ids_train

    
# FOR TEST DATA:

def get_test_data():

    feat_arr_test = []
    ids_test = []

    feat_arr_test_male = []
    ids_test_male = []
    feat_arr_test_female = []
    ids_test_female = []

    root_dir_test = 'TIMIT_Data/speed_perturbation_80fbanks3pitchs/dump/test/deltafalse/' # Root directory for test set
    os.chdir(root_dir_test)

    for i in range(1,21):
        with ReadHelper('scp:' + 'feats' + str(i) + '.scp') as reader:
            for key, numpy_array in reader:
                ids_test.append(key)
                numpy_array = spec_augment(numpy_array)
                
                if numpy_array.shape[0] < 800:
                    numpy_array = pad_sequences(numpy_array.T, maxlen=800, padding='post')
                    numpy_array = numpy_array.T
                
                elif numpy_array.shape[0] > 800:
                    numpy_array = numpy_array[:800]
                    

                # Incorporating gender information as a binary feature:

                if key[0] == 'F': # i.e. for female samples
                    #female_feat_arr_train.append(numpy_array)
                    #feat_arr_train.append(numpy_array)
                    ids_test_female.append(key)
                    feat_arr_test_female.append(np.concatenate((numpy_array, np.array([1]*800).reshape(800,1)), axis=1))
                    feat_arr_test.append(np.concatenate((numpy_array, np.array([1]*800).reshape(800,1)), axis=1))

                elif key[0] == 'M': # i.e. for male samples
                    #male_feat_arr_train.append(numpy_array)
                    #feat_arr_train.append(numpy_array)
                    ids_test_male.append(key)
                    feat_arr_test_male.append(np.concatenate((numpy_array, np.array([0]*800).reshape(800,1)), axis=1))
                    feat_arr_test.append(np.concatenate((numpy_array, np.array([0]*800).reshape(800,1)), axis=1))

                else:
                    print('ERROR! ' + str(key))

    # Coming back to main directory
    os.chdir('../../../../../')

    return feat_arr_test, ids_test, feat_arr_test_female, feat_arr_test_male, ids_test_female, ids_test_male


# FOR VALIDATION DATA:

def get_val_data():

    feat_arr_val = []
    ids_val = []

    root_dir_val = 'TIMIT_Data/speed_perturbation_80fbanks3pitchs/dump/valid/deltafalse/' # Root directory for validation set
    os.chdir(root_dir_val)

    for i in range(1,21):
        with ReadHelper('scp:' + 'feats' + str(i) + '.scp') as reader:
            for key, numpy_array in reader:
                ids_val.append(key)
                numpy_array = spec_augment(numpy_array)
                
                if numpy_array.shape[0] < 800:
                    numpy_array = pad_sequences(numpy_array.T, maxlen=800, padding='post')
                    numpy_array = numpy_array.T
                
                elif numpy_array.shape[0] > 800:
                    numpy_array = numpy_array[:800]
                    

                # Incorporating gender information as a binary feature:

                if key[0] == 'F': # i.e. for female samples
                    #female_feat_arr_train.append(numpy_array)
                    #feat_arr_train.append(numpy_array)
                    feat_arr_val.append(np.concatenate((numpy_array, np.array([1]*800).reshape(800,1)), axis=1))

                elif key[0] == 'M': # i.e. for male samples
                    #male_feat_arr_train.append(numpy_array)
                    #feat_arr_train.append(numpy_array)
                    feat_arr_val.append(np.concatenate((numpy_array, np.array([0]*800).reshape(800,1)), axis=1))
                else:
                    print('ERROR! ' + str(key))

    # Coming back to main directory
    os.chdir('../../../../../')

    return feat_arr_val, ids_val