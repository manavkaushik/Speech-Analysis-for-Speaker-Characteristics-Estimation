import modules_height
import numpy as np
import torch
from torch import nn
from torchmetrics import MeanSquaredError
from torchmetrics import MeanAbsoluteError
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


if __name__ == '__main__':   
    
    # 1. LOAD ENVIRONMENT
    ################ Loading GPU or CPU ###########################################################################
    device = modules_height.pytorch_env()

    ########### Tracking Test MSE & MAE ###########################################################################
    mse_male = MeanSquaredError().to(device)
    mae_male = MeanAbsoluteError().to(device)
    mse_female = MeanSquaredError().to(device)
    mae_female = MeanAbsoluteError().to(device)
    ################################################################################################################

    # 2. EDIT THESE PARAMETERS TO CHANGE THE MODEL HYPER-PARAMETERS 
    params = dict(
        seq_len = 800,                  # Number of timeframes used per sample
        batch_size = 32,                # Number of samples in each batch
        criterion = nn.L1Loss(),        # Loss of BackProp     
        max_epochs = 2,                 # Max Number of Epochs to run the model
        n_features = 84,                # Number of Features per timeframe (84 for this experiment: 80 FBank + 3 Pitch + 1 Gender)
        hidden_size = 64,               # Number of Hidden Units of LSTM
        num_layers = 2,                 # Number of LSTM Layers
        dropout = 0.2,                  # Dropout for LSTM and Dense Layer
        learning_rate = 0.001,          # Learning Rate
        output_size = 1,                # Number of Outputs (1 for height estimation)
        early_stop_patience = 10        # Number of consecutive epochs without any loss reduction after which training stops 
    )
    ################################################################################################################

    # Seeding Everything for Reproducibilty of Results
    seed_everything(7, workers=True)
    csv_logger = CSVLogger('./', name='height_age_mae_multitask', version='0') # Creates a CSV in the folder which contains all the logs (Training + Testing + Validation)

    trainer = Trainer(
        max_epochs=params['max_epochs'],
        logger=csv_logger,              # Logging all the losses, epochs and steps
        gpus= 0,                         # You may change the number of GPUs as per availability 
        #row_log_interval=1,
        progress_bar_refresh_rate=100,   # Number of Epochs after progress is shown regularly
        callbacks=[EarlyStopping(monitor='val_loss', patience= params['early_stop_patience'], mode='min')]  # Early Stopping Callback
    )

    dm = modules_height.Data_Module_height_age(
        seq_len = params['seq_len'],
        batch_size = params['batch_size'], num_workers=4,     # Number of workers for Train_Data_Loader
    )
    ###################################################################################################
    # 3. SELECT THE MODEL TO USE

    model = modules_height.lstm_crossattn_miltitask(
                    n_features = params['n_features'],
                    hidden_size = params['hidden_size'],
                    seq_len = params['seq_len'],
                    batch_size = params['batch_size'],
                    criterion = params['criterion'],
                    num_layers = params['num_layers'],
                    dropout = params['dropout'],
                    learning_rate = params['learning_rate'], 
                    output_size = params['output_size'],
                    mse_female = mse_female, mae_female = mae_female,
                    mse_male = mse_male, mae_male = mae_male
    )
    ####################################################################################################

    # 4. TRAIN AND TEST THE MODEL

    # Training the Model
    trainer.fit(model, dm)            

    # Testing the Model
    trainer.test()                     

    print()
    print('##################### TESTING for HEIGHT ESTIMATION on Test_Set #####################################')
    err_mse_female = mse_female.compute()
    err_mae_female = mae_female.compute()
    err_mse_male = mse_male.compute()
    err_mae_male = mae_male.compute()
    print(f"MSE Height on all Female data: {err_mse_female}")
    print(f"MAE Height on all Female data: {err_mae_female}")
    print(f"MSE Height on all Male data: {err_mse_male}")
    print(f"MAE Height on all Male data: {err_mae_male}")
    print()