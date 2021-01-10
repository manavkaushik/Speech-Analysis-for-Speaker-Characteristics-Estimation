# LABELS LOAD

import numpy as np
import pandas as pd

def get_labels(ids_train, ids_test, ids_val):

    dt = pd.read_csv('TIMIT_Data/labels_data.csv')

    map_dict = {}
    for i in range(dt['ID'].size):
        map_dict[dt['ID'][i]] = [dt['Ht_cm'][i], dt['Age'][i]]

    #print(len(map_dict))

    labels_train = []
    labels_test = []
    labels_val = []
    labels_test_female = []
    labels_test_male = []
    # labels_train_female = []
    # labels_train_male = []
    # labels_val_female = []
    # labels_val_male = []

    labels_train_h = []
    labels_test_h = []
    labels_val_h = []
    labels_train_a = []
    labels_test_a = []
    labels_val_a = []

    for i in ids_train:
        labels_train_h.append(float(map_dict[i[1:5]][0]))
        labels_train_a.append(float(map_dict[i[1:5]][1]))
    #     if i[0] == 'M':
    #         labels_train_male.append([float(map_dict[i[1:5]][0]), float(map_dict[i[1:5]][1])]) 
    #     if i[0] == 'F':
    #         labels_train_female.append([float(map_dict[i[1:5]][0]), float(map_dict[i[1:5]][1])])
        
    # for i in ids_test:
    #     labels_test_h.append(float(map_dict[i[1:5]][0]))
    #     labels_test_a.append(float(map_dict[i[1:5]][1]))
        
    #     if i[0] == 'F':
    #         labels_test_female_a.append(float(map_dict[i[1:5]][1])) 
    #         labels_test_female_h.append(float(map_dict[i[1:5]][0]))
            
    #     if i[0] == 'M':
    #         labels_test_male_a.append(float(map_dict[i[1:5]][1])) 
    #         labels_test_male_h.append(float(map_dict[i[1:5]][0]))
        
        
    for i in ids_test:
        if i[0] == 'M':
            labels_test_male.append([float(map_dict[i[1:5]][0]), float(map_dict[i[1:5]][1])]) 
        if i[0] == 'F':
            labels_test_female.append([float(map_dict[i[1:5]][0]), float(map_dict[i[1:5]][1])])
        
    for i in ids_val:
        labels_val_h.append(float(map_dict[i[1:5]][0]))
        labels_val_a.append(float(map_dict[i[1:5]][1]))
    #     if i[0] == 'M':
    #         labels_val_male.append([float(map_dict[i[1:5]][0]), float(map_dict[i[1:5]][1])]) 
    #     if i[0] == 'F':
    #         labels_val_female.append([float(map_dict[i[1:5]][0]), float(map_dict[i[1:5]][1])])



    for i in range(len(labels_train_h)):
        labels_train.append([float(labels_train_h[i]), float(labels_train_a[i])])
        
    # for i in range(len(labels_test_h)):
    #     labels_test.append([float(labels_test_h[i]), float(labels_test_a[i])])
        
    for i in range(len(labels_val_h)):
        labels_val.append([float(labels_val_h[i]), float(labels_val_a[i])])
        
    # We may avoid scaling age and height as these are already on a very similar scales

    # Saling Age and Height:

    # scaler = MinMaxScaler()
    # data_to_be_scaled = labels_train + labels_val
    # scaler.fit(data_to_be_scaled)
    # scaled_data = scaler.transform(data_to_be_scaled)
    # labels_train = scaled_data[:13170]
    # labels_val = scaled_data[13170:]
        
    # labels_train_male = np.array(labels_train_male).astype(float)
    # labels_train_female = np.array(labels_train_female).astype(float)
    #labels_test = np.array(labels_test).astype(float)
    # labels_test_female = np.array(labels_test_female).astype(float)
    # labels_test_male = np.array(labels_test_male).astype(float)
    # labels_val_male = np.array(labels_val_male).astype(float)
    # labels_val_female = np.array(labels_val_female).astype(float)

    labels_train = np.array(labels_train).astype(float)
    labels_val = np.array(labels_val).astype(float)
    labels_test_female = np.array(labels_test_female).astype(float)
    labels_test_male = np.array(labels_test_male).astype(float)
    # labels_train_a = np.array(labels_train_a)
    # labels_val_a = np.array(labels_val_a)

    return labels_train, labels_val, labels_test_female, labels_test_male
