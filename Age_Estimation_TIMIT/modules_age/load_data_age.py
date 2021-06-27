import numpy as np
import os
from kaldiio import ReadHelper
from torch.utils.data.dataset import Dataset
from .spec_aug import spec_augment 
from .load_labels import get_labels

map_dict = get_labels()

# FOR TRAINING DATA

# TRAIN DATA
class Train_Dataset_age(Dataset):
    # load the dataset
    def __init__(self):
        
        with ReadHelper('scp:./data_83_speed/train/feats.scp') as reader:
            self.dic = { u:d for u,d in reader }
        self.dic_keys = list(self.dic.keys())
        # print(len(self.dic))
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.dic)
 
    # get a row at an index
    def __getitem__(self, idx):
        
        data = self.dic[self.dic_keys[idx]]
        label = self.dic_keys[idx].split('_')[1]
        
        if data.shape[0] < 800:
            data = np.concatenate([data, np.array([[0]*83]*(800-data.shape[0]))])
              
        elif data.shape[0] > 800:
            data = data[:800]
            
        data = spec_augment(data)
            
        if label[0] == 'F':
            data = (np.concatenate((data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif label[0] == 'M':
            data = (np.concatenate((data, np.array([0]*800).reshape(800,1)), axis=1))
        
        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age]
        #print(label)
        #label = self.label_dict[label]
            
        return data, label_age
        
# TEST DATA 
class Test_Dataset_age(Dataset):
    # load the dataset
    def __init__(self):

        with ReadHelper('scp:./data_83_speed/test/feats.scp') as reader:
            self.dic = { u:d for u,d in reader }
        self.dic_keys = list(self.dic.keys())
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.dic)
 
    # get a row at an index
    def __getitem__(self, idx):
        
        data = self.dic[self.dic_keys[idx]]
        label = self.dic_keys[idx].split('_')[0]
        
        if data.shape[0] < 800:
            data = np.concatenate([data, np.array([[0]*83]*(800-data.shape[0]))])
              
        elif data.shape[0] > 800:
            data = data[:800]
            
        if label[0] == 'F':
            data = (np.concatenate((data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif label[0] == 'M':
            data = (np.concatenate((data, np.array([0]*800).reshape(800,1)), axis=1))
            
        if label[0] == 'M':  
            gender = 0
        else:
            gender = 1

        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age, gender]
        #print(label)
        #label = self.label_dict[label]
            
        return data, label_age, gender

        
# VALIDATION DATA
class Val_Dataset_age(Dataset):
    # load the dataset
    def __init__(self):
        
        with ReadHelper('scp:./data_83_speed/valid/feats.scp') as reader:
            self.dic = { u:d for u,d in reader }
        self.dic_keys = list(self.dic.keys())
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.dic)
 
    # get a row at an index
    def __getitem__(self, idx):
        
        data = self.dic[self.dic_keys[idx]]
        label = self.dic_keys[idx].split('_')[0]
        
        if data.shape[0] < 800:
            data = np.concatenate([data, np.array([[0]*83]*(800-data.shape[0]))])
              
        elif data.shape[0] > 800:
            data = data[:800]
            
        if label[0] == 'F':
            data = (np.concatenate((data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif label[0] == 'M':
            data = (np.concatenate((data, np.array([0]*800).reshape(800,1)), axis=1))
        
        label_ht = float(map_dict[label[1:5]][0])
        label_age = float(map_dict[label[1:5]][1])
        labels = [label_ht, label_age]
        #print(label)
        #label = self.label_dict[label]
            
        return data, label_age