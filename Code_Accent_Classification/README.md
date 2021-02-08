# Code for Accent Classification

## Overview of the Directory

This folder contains the code implementation of different models for AGE and HEIGHT estimation for 'Accented speech recognition workshop data’ from DataTang for INTERSPEECH 2020 </br>

The directory also contains augmented features for Training, Validation and Testing Sets. </br>
Number of Accents: 8 (each ~20 hrs)
- American
- British
- Chinese
- Indian
- Japanese
- Korean
- Portuguese
- Russian

The features here include:</br>
- 80 Filter Bank Energy features
- 3 Pitch features </br>

</br>
We have used CMVN normalization as a pre-processing step on the input features.
</br></br>

 

As of now, the following six models have been added in the package here:</br>

```
- LSTM + Attention Model
- LSTM + Cross Attention (proposed by us) Model
- BiLSTM + Cross Attention + Focal Loss Model.
- LSTM + Cross Attention + Multi-Task Model (Accent & Gender prediction)
- LSTM + Cross Attention + Multi-Task + Focal Loss Model (Accent & Gender prediction)
- LSTM + Cross Attention + Gender_Pre_Training Model
```
</br></br>

## Running the Package:

1. Please ensure that the follwing dependencies are installed in you system:</br>
```
sklearn
kaldiio
keras
tensorflow
numpy
pandas
pickle
```
</br></br>

2. Clone or Download this directory (i.e. Code_Accent_Classification). </br></br>

3. Excecute the `run.py` file (make sure that your current working directory is 'Code_Accent_Classification'):
```
$ python run.py
```

(You may change the model to be used in the `run.py`. If you do not make any changes, `BiLSTM + Cross Attention + Focal Loss Model` shall run by default).


## For Further Details:

For gaining further insight of our work, you may refer the following report compiled by us on this topic:
