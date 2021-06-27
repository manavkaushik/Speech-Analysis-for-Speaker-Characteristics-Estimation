# Models for Age Estimation:

This document is to compile the summary of all the models for age estimation estimation using TIMIT dataset. </br>
We predominantly use two kinds of features for these models:
- **Filter Bank**: 80 FBank + 3 Pitch + 1 Binary_Gender (Features_Dimension: 83)
- **Wav2Vec2**: Features extracted from pre-trained Wav2Vec2 model (Features_Dimension: 768)

Moreover, we use 3 data augmentations for our data:
- **CMVN**: Cepstral mean and variance normalization for FBank features
- **Speed Perturbation**: Triple the training data using 0.9x and 1.1x speed perturbed data.
- **Spectral Augmentation**: SpecAugment to randomly mask 15%-25% for better generalization and robustness.

</br></br>

## **Model_1**:

- **Model**: `LSTM + Cross_Attention + MSE_Loss/ MAE_Loss` | FBank Features | Age Estimation
- **Model Description**: The model uses FBank features for only age estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Squared Error (MSE)` or `Mean Absolute Error (MAE)` loss and `Adam` optimizer. We use a patience of `10` epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are used to gauge the performance of the model on the `test_set` of TIMIT for age estimation. The `batch_size` used is `32` with an initial `learning rate` of `0.001`.

- **Model Architecture**: </br>
<img src="/Age_Estimation_TIMIT/imgs/age.png" width="300">

- **Results**: </br>

|S. No. | Model                              | Features              | Loss                             | Gender  | Age RMSE   | Age MAE  |
| ----- | ---------------------------------- | --------------------- | -------------------------------- | ------- | ---------- | -------- |
| 1.    | LSTM + Cross_att (SingleTask)      | Filter Bank & Pitch   | Mean Squared Error (MSE)         | Male    | 7.80       | 5.52     |
|       |                                    |                       |                                  | Female  | 8.70       | 6.05     |
| 2.    | LSTM + Cross_att (SingleTask)      | Filter Bank & Pitch   | Mean Asbsolute Error (MAE)       | Male    | 8.15       | 5.39     |
|       |                                    |                       |                                  | Female  | 9.63       | 6.30     |

</br></br>

## **Model_2**:

- **Model**: `LSTM + Cross_Attention + MSE_Loss + MultiTask Learning` | FBank Features | MultiTask Estimation (both age & height)
- **Model Description**: The model uses FBank features for only height estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Squared Error (MSE)` loss and `Adam` optimizer for both age and height estimation with `height_loss` given the same weight as that of `age_loss`. 
We use a patience of `10` epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the `test_set` for age estimation. The `batch_size` used is `32` with an initial `learning rate` of `0.001`.

- **Model Architecture**: </br>
<img src="/Age_Estimation_TIMIT/imgs/height_age_mse.png" width="500">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Age RMSE   | Age MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ---------- | -------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | Mean Squared Error (MSE)         | Male    | 7.90       | 5.62     |
|       | MultiTask             |                       |                                  | Female  | 8.10       | 5.91     |

</br></br>

## **Model_3**:

- **Model**: `LSTM + Cross_Attention + Triplet & MSE_Loss` | FBank Features | Age Estimation
- **Model Description**: The model uses FBank features for only age estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Squared Error (MSE)` loss combined with a `Triplet Loss`, used to train the `embeddings` obtained right after the `cross_attention layer`. `Triplet loss` is given one-third the weighatge in total loss while `MSE` is given two-thirds. `Adam` is used the optimizer. The age labels are quantized and classified into groups of 5years for Triplet Loss (i.e. age labels from 20-25years in `class_0`, 25-30years in `class_1` and so on, giving us a total of `12` classes). 
We use a patience of `10` epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for age estimation. The `batch_size` used is `32` with an initial `learning rate` of `0.001`.


- **Model Architecture**: </br>
<img src="/Age_Estimation_TIMIT/imgs/age_triplet.png" width="400">

- **Results**: State-of-the-Art </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Age RMSE   | Age MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ---------- | -------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | MSE + Triplet Loss               | Male    | 7.18       | 5.03     |
|       | SingleTask            |                       |                                  | Female  | 6.92       | 4.95     |

</br></br>

## **Model_4**:

- **Model**: `LSTM + Cross_Attention + Center & MSE_Loss` | FBank Features | Age Estimation
- **Model Description**: The model uses FBank features for only age estimation using standart `LSTM + Cross_Attnetion + Dense Layer` and is trained using a 
`Mean Squared Error (MSE)` loss combined with a `Center Loss`, used to train the `embeddings` obtained right after the `cross_attention layer`. `Center loss` is given one-third the weighatge in total loss while `MSE` is given two-thirds. `Adam` is used the optimizer. The age labels are quantized and classified into groups of 5years for Center Loss (i.e. age labels from 20-25years in `class_0`, 25-30years in `class_1` and so on, giving us a total of `12` classes). 
We use a patience of `10` epochs before early stopping the model based on Validation Loss. Finally, `MSE` and `MAE` metrics are 
used to gauge the performance of the model on the test_set for age estimation. The `batch_size` used is `32` with an initial `learning rate` of `0.001`.


- **Model Architecture**: </br>
<img src="/Age_Estimation_TIMIT/imgs/age_center.png" width="400">

- **Results**: </br>

|S. No. | Model                 | Features              | Loss                             | Gender  | Age RMSE   | Age MAE  |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ---------- | -------- |
| 1.    | LSTM + Cross_att      | Filter Bank & Pitch   | MSE + Center Loss                | Male    | 7.09       | 5.06     |
|       | SingleTask            |                       |                                  | Female  | 7.33       | 5.25     |

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

2. Change your current working directory to `Age_Estimation_TIMIT`. 

3. Run the model_run file which you wish to reproduce. For example: </br>
```
$ python age_triplet_mse_run.py
```

## Other instructions:

- You may change the hyper-parameters such as the `batch_size`, `max_epochs`, `early_stopping_patience`, `learning_rate`, `num_layers`, `loss_criterion`, etc. in the run.py file of any model.
- Please note that the if you are not using a GPU for processing, change the hyper-parameter of `gpu` in the `trainer` function (in the run.py files) to `0`.
