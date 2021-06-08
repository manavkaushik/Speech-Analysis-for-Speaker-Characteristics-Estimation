# Code Package: Tensorflow

## Overview of the Directory

This folder contains the code implementation of different models for AGE and HEIGHT estimation for TIMIT Dataset. </br>
The directory also contains augmented features for Training, Validation and Testing Sets of TIMIT.
The features here include:</br>
- 80 Filter Bank Energy features
- 3 Pitch features </br>

(Gender has also been used as a binary features and is taken up during the model training itself by the code)</br></br>

We have used Speed Perturbation and enhanced the dataset size (for training) three-folds (at 1x speed, 1.1x speed and 0.9x speed).

As of now, the following three models have been added in the package here:</br>
- Base LSTM Model
- Attention Model (from the Literature)
- Cross Attention Model (proposed by us).</br></br>

## Running the Package:

1. Please ensure that the follwing dependencies are installed in you system:</br>
```
sklearn
kaldiio
keras
tensorflow
numpy
pandas
```
</br></br>

2. Clone or Download this directory (i.e. Code). </br></br>

3. Excecute the `run.py` file (make sure that your current working directory is 'Code'):
```
$ python run.py
```

(You may change the model to be used in the `run.py`. If you do not make any changes, Cross Attention Model shall run by default).

## Attention Tracking:

The code for attention has been compiled keeping in mind this task. You may access the attention weights anytime after training the model. The function (`attention`) returns the attention weights for all the phones used.

## For Further Details:

For gaining further insight of our work, you may refer the following research paper written by us on this topic:
