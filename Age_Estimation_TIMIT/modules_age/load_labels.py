import numpy as np
import pandas as pd


def get_labels():
    dt = pd.read_excel("./data_83_speed/data_cleaned.xlsx") 
    #print(dt.head())
    
    map_dict = {}
    
    for i in range(dt['ID'].size):
        map_dict[dt['ID'][i]] = [dt['Ht_cm'][i], dt['Age'][i], dt['Sex'][i]]

    return map_dict
