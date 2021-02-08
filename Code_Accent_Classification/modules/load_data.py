import numpy as np
import os
from kaldiio import ReadHelper
#from .spec_aug import spec_augment 
from keras.preprocessing.sequence import pad_sequences


# FOR TRAINING DATA

def data_gen_train():
  
  #os.chdir('../')
  root_dir_train = 'Accent_Data/train/'
  print(os.getcwd())
  os.chdir(root_dir_train)
  print(os.getcwd())
  for i in range(1,3):
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
                            
              yield numpy_array, label
              
def data_gen_test():
  
  os.chdir('../../')
  print('FOR TEST DIR..............')
  root_dir_test = 'Accent_Data/test/'
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
              
              yield numpy_array, label
              
def data_gen_val():

    root_dir_val = 'Accent_Data/dev/'

    os.chdir('../../')
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
                
                yield numpy_array, label      



# def data_batch_pipe_train():

#     ds_series_train = tf.data.Dataset.from_generator(
#         data_gen_train, 
#         output_types=(tf.float32, tf.float32),
#         output_shapes=((1000, 83), (8)))

#     print('Train Data: {}'.format(ds_series_train))
#     ds_series_train_batch = ds_series_train.shuffle(125555).padded_batch(64)

#     return ds_series_train_batch
    
    

# def data_batch_pipe_test():

#     ds_series_test = tf.data.Dataset.from_generator(
#         data_gen_test, 
#         output_types=(tf.float32, tf.float32),
#         output_shapes=((1000, 83), (8)))

#     print('Test Data: {}'.format(ds_series_test))
#     ds_series_test_batch = ds_series_test.shuffle(15000).padded_batch(64)

#     return ds_series_test_batch



# def data_batch_pipe_val():
    
#     ds_series_val = tf.data.Dataset.from_generator(
#         data_gen_val, 
#         output_types=(tf.float32, tf.float32),
#         output_shapes=((1000, 83), (8)))

#     print('Val Data: {}'.format(ds_series_val))
#     ds_series_val_batch = ds_series_val.shuffle(11988).padded_batch(64)

#     return ds_series_val_batch
