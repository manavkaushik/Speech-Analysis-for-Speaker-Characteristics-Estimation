import numpy as np
import os
import pickle
from kaldiio import ReadHelper
#from .spec_aug import spec_augment 
from keras.preprocessing.sequence import pad_sequences


# FOR TRAINING DATA

def data_gen_train_gender():

  ########### Training Data Gender Labels ################

    root_dir_train = 'Data/train/'
    dict_gender_train = {}
    count = 0

    f = open(root_dir_train + 'utt2sex', "r")
    for line in f:
        
        splt_1 = line.split(' ')
        splt_2 = splt_1[0].split('-')

        meta = ([splt_2[1], int(splt_1[1])])
        
        dict_gender_train[splt_1[0]] = meta
        
    f.close()
  
    #os.chdir('../')
    print(os.getcwd())
    os.chdir(root_dir_train)
    print(os.getcwd())
    for i in range(1,33):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                    #max_len = numpy_array.shape[0]
        
                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])  
                            
                yield(numpy_array, {'accent': label, 'gender': np.array([float(dict_gender_train[key][1])])} )
              
def data_gen_test_gender():
  
  os.chdir('../../')
  print('FOR TEST DIR..............')
  root_dir_test = 'Data/test/'
  #print(os.getcwd())
  os.chdir(root_dir_test)
  print(os.getcwd())

  for i in range(1,9):
      with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
          for key, numpy_array in reader:
              #ids_train.append(key.split('-')[1])
              #if numpy_array.shape[0] > max_len:
                    #max_len = numpy_array.shape[0]
      
              if numpy_array.shape[0] < 1000:
                  numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
              
              elif numpy_array.shape[0] > 1000:
                  numpy_array = numpy_array[:1000]
                  #n = n+1
                  
              if key.split('-')[1] == 'AMERICAN':
                  label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                  
              elif key.split('-')[1] == 'BRITISH':
                  label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                  
              elif key.split('-')[1] == 'CHINESE':
                  label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
              
              elif key.split('-')[1] == 'INDIAN':
                  label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                  
              elif key.split('-')[1] == 'JAPANESE':
                  label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                  
              elif key.split('-')[1] == 'KOREAN':
                  label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                  
              elif key.split('-')[1] == 'PORTUGUESE':
                  label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                  
              elif key.split('-')[1] == 'RUSSIAN':
                  label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
              
              yield(numpy_array, {'accent': label, 'gender': np.array([float(1)])} )
              
def data_gen_val_gender():

    ########### Validation Data Gender Labels ################

    os.chdir('../../')
    root_dir_val = 'Data/dev/'
    dict_gender_dev = {}
    count = 0

    f = open(root_dir_val + 'utt2sex', "r")
    for line in f:
        
        splt_1 = line.split(' ')
        splt_2 = splt_1[0].split('-')

        meta = ([splt_2[1], int(splt_1[1])])
        
        dict_gender_dev[splt_1[0]] = meta
            
    f.close()

    os.chdir(root_dir_val)
    for i in range(1,9):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                        #max_len = numpy_array.shape[0]
        
                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                
                
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
                
                yield(numpy_array, {'accent': label, 'gender': np.array([float(dict_gender_dev[key][1])])} )     









############################### FOR GENDER PRETRAINING MODEL #################################################


def data_gen_train_0():
  
    root_dir_train = 'Data/train/'
    os.chdir(root_dir_train)

    dict_gender_train = {}


    f = open('utt2sex', "r")
    for line in f:
        
        splt_1 = line.split(' ')
        splt_2 = splt_1[0].split('-')

        meta = ([splt_2[1], int(splt_1[1])])
        
        dict_gender_train[splt_1[0]] = meta
        
    f.close()
  
  
    for i in range(1,33):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                    #max_len = numpy_array.shape[0]
        
                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])  
                            
                #yield(numpy_array, {'accent': label, 'gender': np.array([float(dict_gender_train[key][1])])} )
                
                if dict_gender_train[key][1] == 0:
                    yield numpy_array, label # , np.array([float(dict_gender_train[key][1])])
                  




def data_gen_train_1():
  
    root_dir_train = 'Data/train/'
    os.chdir(root_dir_train)
    #print(os.getcwd())
    dict_gender_train = {}


    f = open('utt2sex', "r")
    for line in f:
        
        splt_1 = line.split(' ')
        splt_2 = splt_1[0].split('-')

        meta = ([splt_2[1], int(splt_1[1])])
        
        dict_gender_train[splt_1[0]] = meta
        
    f.close()

    for i in range(1,33):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                    #max_len = numpy_array.shape[0]

                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])  
                            
                #yield(numpy_array, {'accent': label, 'gender': np.array([float(dict_gender_train[key][1])])} )
                
                if dict_gender_train[key][1] == 1:
                    yield numpy_array, label # , np.array([float(dict_gender_train[key][1])])

  
            
def data_gen_test_1():
  
    os.chdir('../../')
    root_dir_test = 'Data/test/'
    print('FOR TEST DIR..............')
    print(os.getcwd())
    os.chdir(root_dir_test)
    print(os.getcwd())

    file_to_read = open("test_dict", "rb")
    test_dict = pickle.load(file_to_read)

    for i in range(1,9):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                    #max_len = numpy_array.shape[0]
        
                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                    
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
                
                #yield(numpy_array, {'accent': label, 'gender': np.array([float(1)])} )
                if test_dict[key] == 1:
                    yield numpy_array, label # , np.array([float(1)])



           
def data_gen_test_0():
  
    os.chdir('../../')
    root_dir_test = 'Data/test/'
    print('FOR TEST DIR..............')
    print(os.getcwd())
    os.chdir(root_dir_test)
    print(os.getcwd())
    file_to_read = open("test_dict", "rb")
    test_dict = pickle.load(file_to_read)

    for i in range(1,9):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                    #max_len = numpy_array.shape[0]
        
                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                    
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
                
                #yield(numpy_array, {'accent': label, 'gender': np.array([float(1)])} )
                if test_dict[key] == 0:
                    yield numpy_array, label # , np.array([float(1)])          

      
def data_gen_val_0():

    root_dir_val = 'Data/dev/'
    os.chdir('../../')
    os.chdir(root_dir_val)
    dict_gender_dev = {}

    f = open('utt2sex', "r")
    for line in f:
        
        splt_1 = line.split(' ')
        splt_2 = splt_1[0].split('-')

        meta = ([splt_2[1], int(splt_1[1])])
        
        dict_gender_dev[splt_1[0]] = meta
            
    f.close() 

    
    for i in range(1,9):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                        #max_len = numpy_array.shape[0]

                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                
                
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
                    
                    
                #yield(numpy_array, {'accent': label, 'gender': np.array([float(dict_gender_dev[key][1])])} )
                
                if dict_gender_dev[key][1] == 0:
                    yield numpy_array, label # , np.array([float(dict_gender_dev[key][1])])
  

              
        
def data_gen_val_1():

    root_dir_val = 'Data/dev/'
    os.chdir('../../')
    os.chdir(root_dir_val)

    dict_gender_dev = {}

    f = open('utt2sex', "r")
    for line in f:
        
        splt_1 = line.split(' ')
        splt_2 = splt_1[0].split('-')

        meta = ([splt_2[1], int(splt_1[1])])
        
        dict_gender_dev[splt_1[0]] = meta
            
    f.close() 

    for i in range(1,9):
        with ReadHelper("scp:" + "feats" + str(i) + ".scp") as reader:
            for key, numpy_array in reader:
                #ids_train.append(key.split('-')[1])
                #if numpy_array.shape[0] > max_len:
                        #max_len = numpy_array.shape[0]

                if numpy_array.shape[0] < 1000:
                    numpy_array = np.concatenate([numpy_array, np.array([[0]*83]*(1000-numpy_array.shape[0]))])
                
                elif numpy_array.shape[0] > 1000:
                    numpy_array = numpy_array[:1000]
                    #n = n+1
                
                
                if key.split('-')[1] == 'AMERICAN':
                    label = np.array([1., 0., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'BRITISH':
                    label = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'CHINESE':
                    label = np.array([0., 0., 1., 0., 0., 0., 0., 0.])
                
                elif key.split('-')[1] == 'INDIAN':
                    label = np.array([0., 0., 0., 1., 0., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'JAPANESE':
                    label = np.array([0., 0., 0., 0., 1., 0., 0., 0.])
                    
                elif key.split('-')[1] == 'KOREAN':
                    label = np.array([0., 0., 0., 0., 0., 1., 0., 0.])
                    
                elif key.split('-')[1] == 'PORTUGUESE':
                    label = np.array([0., 0., 0., 0., 0., 0., 1., 0.])   
                    
                elif key.split('-')[1] == 'RUSSIAN':
                    label = np.array([0., 0., 0., 0., 0., 0., 0., 1.])
                    
                    
                #yield(numpy_array, {'accent': label, 'gender': np.array([float(dict_gender_dev[key][1])])} )
                
                if dict_gender_dev[key][1] == 1:
                    yield numpy_array, label # , np.array([float(dict_gender_dev[key][1])])
