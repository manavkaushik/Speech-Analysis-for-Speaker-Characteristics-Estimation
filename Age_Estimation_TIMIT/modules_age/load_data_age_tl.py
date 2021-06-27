import numpy as np
import os
import math, random
from kaldiio import ReadHelper
from torch.utils.data.dataset import Dataset
from .spec_aug import spec_augment 
from .load_labels import get_labels

map_dict = get_labels()

# FOR TRAINING DATA

# TRAIN DATA
class Train_Dataset_age_tl(Dataset):
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
        
        anchor_data = self.dic[self.dic_keys[idx]]
        anchor_label = self.dic_keys[idx].split('_')[1]
        n_anchor = math.floor(map_dict[anchor_label[1:5]][1] / 5) - 4
        
        positive_list = []
        negative_list = []
        for i in self.dic_keys:
            
            j = i.split('_')[1]
            if math.floor(map_dict[j[1:5]][1] / 5)-4 == n_anchor:
                positive_list.append(i)
            else:
                negative_list.append(i)

        positive_label = random.choice(positive_list)
        positive_data = self.dic[positive_label]


        negative_label = random.choice(negative_list)
        negative_data = self.dic[negative_label]

        if anchor_data.shape[0] < 800:
            anchor_data = np.concatenate([anchor_data, np.array([[0]*83]*(800-anchor_data.shape[0]))])
              
        elif anchor_data.shape[0] > 800:
            anchor_data = anchor_data[:800]
            
        if anchor_label[0] == 'F':
            anchor_data = (np.concatenate((anchor_data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif anchor_label[0] == 'M':
            anchor_data = (np.concatenate((anchor_data, np.array([0]*800).reshape(800,1)), axis=1))

#####################

        if positive_data.shape[0] < 800:
            positive_data = np.concatenate([positive_data, np.array([[0]*83]*(800-positive_data.shape[0]))])
              
        elif positive_data.shape[0] > 800:
            positive_data = positive_data[:800]
            
        if positive_label.split('_')[1][0] == 'F':
            positive_data = (np.concatenate((positive_data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif positive_label.split('_')[1][0] == 'M':
            positive_data = (np.concatenate((positive_data, np.array([0]*800).reshape(800,1)), axis=1))

#####################

        if negative_data.shape[0] < 800:
            negative_data = np.concatenate([negative_data, np.array([[0]*83]*(800-negative_data.shape[0]))])
              
        elif negative_data.shape[0] > 800:
            negative_data = negative_data[:800]
            
        if negative_label.split('_')[1][0] == 'F':
            negative_data = (np.concatenate((negative_data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif negative_label.split('_')[1][0] == 'M':
            negative_data = (np.concatenate((negative_data, np.array([0]*800).reshape(800,1)), axis=1))
            
        #print(data.shape)
        
        anchor_data, positive_data, negative_data = spec_augment(anchor_data), spec_augment(positive_data), spec_augment(negative_data)
        
        label = float(map_dict[anchor_label[1:5]][1])
        #print(label)
        #label = self.label_dict[label]
            
        return anchor_data, positive_data, negative_data, label
        
# TEST DATA 
class Test_Dataset_age_tl(Dataset):
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
class Val_Dataset_age_tl(Dataset):
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
        
        anchor_data = self.dic[self.dic_keys[idx]]
        anchor_label = self.dic_keys[idx].split('_')[0]
        n_anchor = math.floor(map_dict[anchor_label[1:5]][1] / 5) - 4
        
        positive_list = []
        negative_list = []
        for i in self.dic_keys:
            
            j = i.split('_')[0]
            if math.floor(map_dict[j[1:5]][1] / 5)-4 == n_anchor:
                positive_list.append(i)
            else:
                negative_list.append(i)

        positive_label = random.choice(positive_list)
        positive_data = self.dic[positive_label]


        negative_label = random.choice(negative_list)
        negative_data = self.dic[negative_label]

        if anchor_data.shape[0] < 800:
            anchor_data = np.concatenate([anchor_data, np.array([[0]*83]*(800-anchor_data.shape[0]))])
              
        elif anchor_data.shape[0] > 800:
            anchor_data = anchor_data[:800]
            
        if anchor_label[0] == 'F':
            anchor_data = (np.concatenate((anchor_data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif anchor_label[0] == 'M':
            anchor_data = (np.concatenate((anchor_data, np.array([0]*800).reshape(800,1)), axis=1))

#####################

        if positive_data.shape[0] < 800:
            positive_data = np.concatenate([positive_data, np.array([[0]*83]*(800-positive_data.shape[0]))])
              
        elif positive_data.shape[0] > 800:
            positive_data = positive_data[:800]
            
        if positive_label.split('_')[0][0] == 'F':
            positive_data = (np.concatenate((positive_data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif positive_label.split('_')[0][0] == 'M':
            positive_data = (np.concatenate((positive_data, np.array([0]*800).reshape(800,1)), axis=1))

#####################

        if negative_data.shape[0] < 800:
            negative_data = np.concatenate([negative_data, np.array([[0]*83]*(800-negative_data.shape[0]))])
              
        elif negative_data.shape[0] > 800:
            negative_data = negative_data[:800]
            
        if negative_label.split('_')[0][0] == 'F':
            negative_data = (np.concatenate((negative_data, np.array([1]*800).reshape(800,1)), axis=1))
                
        elif negative_label.split('_')[0][0] == 'M':
            negative_data = (np.concatenate((negative_data, np.array([0]*800).reshape(800,1)), axis=1))
            
        #print(data.shape)
        
        #anchor_data, positive_data, negative_data = spec_augment(anchor_data), spec_augment(positive_data), spec_augment(negative_data)
        
        label = float(map_dict[anchor_label[1:5]][1])
        #print(label)
        #label = self.label_dict[label]
            
        return anchor_data, positive_data, negative_data, label