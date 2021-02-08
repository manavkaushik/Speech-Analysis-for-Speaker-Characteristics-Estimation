import modules



# SELECTING HYPERPARAMETERS #

max_epochs = 100            ## No. of maximum epochs to be trained
batch_size = 64             ## Batch size for training
early_stop_patience = 10    ## No. of consecutive epochs after which the models stops if no reduction in VALIDATION LOSS is observed 
label_smoothing = 0.0       ## Label smoothing factor for Categorical_Crossentropy loss function


# SELECTING THE MODEL #
# Choose any one of the following model and comment out the others:

model = modules.bilstm_cross_attn_focal_model()
# model = modules.lstm_att_model()
# model = modules.lstm_cross_att_model()
# model = modules.lstm_cross_att_multitask_model()
# model = modules.lstm_cross_att_focal_multitask_model()
# model = modules.lstm_cross_att_gender_pretrain_model()




