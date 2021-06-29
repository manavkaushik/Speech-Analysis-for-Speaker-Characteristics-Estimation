# Models:

This document is to compile the summary of all the models for height and age estimation using TIMIT dataset. </br>
We predominantly use two kinds of features for these models:
- **Filter Bank**: 80 FBank + 3 Pitch + 1 Binary_Gender (Features_Dimension: 83)
- **Wav2Vec2**: Features extracted from pre-trained Wav2Vec2 model (Features_Dimension: 768)

Moreover, we use 3 data augmentations for our data:
- **CMVN**: Cepstral mean and variance normalization for FBank features
- **Speed Perturbation**: Triple the training data using 0.9x and 1.1x speed perturbed data.
- **Spectral Augmentation**: SpecAugment to randomly mask 15%-25% for better generalization and robustness.

</br></br>

## **Model_1**:

- **Model**: `LSTM + Cross_Attention + MSE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Squared Error (MSE)` loss and `Adam` optimizer. We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are used to gauge the performance of the model on the `test_set` for height estimation. The `batch_size` used is 32.

- **Model Architecture**: </br>
<img src="/Height_Estimation_TIMIT/imgs/height_mse.png" width="300">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Squared Error (MSE)         | Male    | 6.98          | 5.30        |
|       | SingleTask            |                       |                                  | Female  | 6.50          | 5.22        |
 
</br></br>
 
 ## **Model_2**:

- **Model**: `LSTM + Cross_Attention + MAE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Absilute Error (MAE)` loss and `Adam` optimizer. We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the `test_set` for height estimation. The `batch_size` used is 32.

- **Model Architecture**: </br>
<img src="/Height_Estimation_TIMIT/imgs/height_mae.png" width="300">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Absolute Error (MAE)        | Male    | 6.98          | 5.31        |
|       | SingleTask            |                       |                                  | Female  | 6.43          | 5.07        |

</br></br>

## **Model_3**:

- **Model**: `LSTM + Cross_Attention + MSE_Loss` | FBank Features | MultiTask Estimation (both age & height)
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Squared Error (MSE)` loss and `Adam` optimizer for both age and height estimation with `height_loss` given twice the weight as comapred to `age_loss`. 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the `test_set` for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="/Height_Estimation_TIMIT/imgs/height_age_mse.png" width="500">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Squared Error (MSE)         | Male    | 6.95          | 5.26        |
|       | MultiTask             |                       |                                  | Female  | 6.44          | 5.15        |

</br></br>

## **Model_4**:

- **Model**: `LSTM + Cross_Attention + MAE_Loss` | FBank Features | MultiTask Estimation (both age & height)
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Absolute Error (MAE)` loss and `Adam` optimizer for both age and height estimation with `height_loss` given twice the weight as comapred to `age_loss`. 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="/Height_Estimation_TIMIT/imgs/height_age_mae.png" width="500">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Absolute Error (MAE)        | Male    | 7.14          | 5.54        |
|       | MultiTask             |                       |                                  | Female  | 6.48          | 4.95        |

</br></br>

## **Model_5**:

- **Model**: `LSTM + Cross_Attention + Triplet & MSE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Squared Error (MAE)` loss combined with a `Triplet Loss`, used to train the `embeddings` obtained right after the `cross_attention layer`. `Triplet loss` is given one-third the 
weighatge in total loss while `MSE` is given two-thirds. `Adam` is used the optimizer. The height labels are quantized and classified into groups of 5cms for Triplet Loss (i.e. height labels from 140-145cm in class_0, 145-150cm in class_1 and so on, giving us a total of 13 classes). 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="/Height_Estimation_TIMIT/imgs/height_triplet.png" width="400">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | MSE + Triplet Loss               | Male    | 6.92          | 5.26        |
|       | SingleTask            |                       |                                  | Female  | 6.24          | 4.95        |

</br></br>

## **Model_6**:

- **Model**: `LSTM + Cross_Attention + Center & MSE_Loss` | FBank Features | Height Estimation
- **Model Description**: The model uses FBank features for only height estimation using standart LSTM + Cross_Attnetion + Dense Layer and is trained using a 
`Mean Squared Error (MAE)` loss combined with a `Center Loss`, used to train the `embeddings` obtained right after the `cross_attention layer`. `Center loss` is given one-third the 
weighatge in total loss while `MSE` is given two-thirds. `Adam` is used the optimizer. The height labels are quantized and classified into groups of 5cms for Center Loss (i.e. height labels from 140-145cm in `class_0`, 145-150cm in `class_1` and so on, giving us a total of 13 classes). 
We use a patience of 10 epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for height estimation. The `batch_size` used is of 32 samples.

- **Model Architecture**: </br>
<img src="/Height_Estimation_TIMIT/imgs/height_center.png" width="400">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Height RMSE   | Height MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------------- | ----------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | MSE + Center Loss                | Male    | 6.96          | 5.27        |
|       | SingleTask            |                       |                                  | Female  | 6.47          | 5.18        |

</br></br>

# How to reproduce the models?

## Requirements:

```
- numpy
- pandas
- torch
- pytorch_lightning
- torchmetrics
```

## 3 Steps to run the model:

1. Clone or download this repository into your system.

2. Change your current working directory to `Height_Estimation_TIMIT`. 

3. Run the model_run file which you wish to reproduce. For example: </br>
```
$ python height_triplet_mse_run.py
```

## Other instructions:

- You may change the hyper-parameters such as the `batch_size`, `max_epochs`, `early_stopping_patience`, `learning_rate`, `num_layers`, `loss_criterion`, etc. in the run.py file of any model.
- Please note that the if you are not using a GPU for processing, change the hyper-parameter of `gpu` in the `trainer` function (in the run.py files) to `0`.
