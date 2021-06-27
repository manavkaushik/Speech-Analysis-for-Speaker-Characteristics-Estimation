from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .load_data_age import Train_Dataset_age
from .load_data_age import Test_Dataset_age
from .load_data_age import Val_Dataset_age


# Create a Pytorch Lightning Data Module to be used for training, validation and testing #

class Data_Module_age(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading 
      and processing work in one place.
    '''
    
    def __init__(self, seq_len = 800, batch_size = 32, num_workers=0):
        super().__init__()
        # self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        '''
        You may do any pre-processing for your data here.
        '''
        pass
    
    def train_dataloader(self):
        train_dataset = Train_Dataset_age()
        train_loader = DataLoader(train_dataset, 
                                  batch_size = self.batch_size, 
                                  shuffle = True, 
                                  num_workers = self.num_workers)
        
        return train_loader

    def val_dataloader(self):
        val_dataset = Val_Dataset_age()
        val_loader = DataLoader(val_dataset, 
                                batch_size = self.batch_size, 
                                shuffle = False, 
                                num_workers = 1)

        return val_loader

    def test_dataloader(self):
        test_dataset = Test_Dataset_age()
        test_loader = DataLoader(test_dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = False, 
                                 num_workers = 1)

        return test_loader