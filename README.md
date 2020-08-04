# Speech-Analysis-for-Speaker-Characteristics-Estimation

## Table of Content:

- **Introduction**
- **Topics to be Covered**
- **Age:**
  - [x] Recent Publications
  - [x] Datasets
  - [x] Comparison of different techniques
  - [ ] Resources & Code
- **Height:**
  - [x] Recent Publications
  - [x] Datasets
  - [x] Comparison of different techniques
  - [ ] Resources & Code
- **Accent:**
  - [x] Recent Publications
  - [x] Datasets
  - [x] Comparison of different techniques
  - [ ] Resources & Code
- **Miscellaneous**

</br></br></br>

## Introduction:

Hello everyone! </br>
I, under the guidance of my thesis supervisor: [Prof. Chng Eng Siong](https://www.ntu.edu.sg/home/aseschng/intro1.html) (School of Computing, Nanyang Technological University Singapore), am compiling this repository with a sincere hope of benefitting the community from the same. </br>
This is a one-stop-repository for most of the recent works and developments in the domain of 'speech analysis for speaker characteristic recignition and profiling'. </br>
I shall try to cover as much content as I can on the said topic including my own work. </br>

## Topics to be Covered:

For the time being, we shall be covering predominantly three aspects of speaker profiling:
1. Age
2. Height
3. Accent </br>

I shall be adding all the recent publications along with their respective analysis and a brief comparisons of their results with other works separately for Age, Height, and Accent. Moreover, I shall be briefing about the popular datasets which are being used in the literature over the years of these purposes. Finally, I shall be sharing all the resources and codes that I compile or come across for our purpose. </br>

## Age:

### Publications:

I am citing some of the recent of works of literature that I have come across and found useful:

1. [Combining Five Acoustic Level Modeling Methods for Automatic Speaker Age and Gender Recognition](https://www.researchgate.net/publication/221483984_Combining_five_acoustic_level_modeling_methods_for_automatic_speaker_age_and_gender_recognition)
2. [Automatic speaker age and gender recognition using acoustic and prosodic level information fusion](https://sail.usc.edu/publications/files/Li-AgeGender-CSL-2013.pdf)
3. [Age and Gender Recognition Based on Multiple Systems - Early vs. Late Fusion](https://www.researchgate.net/publication/221484470_Age_and_gender_recognition_based_on_multiple_systems_-_early_vs_late_fusion)
4. [A DEEP NEURAL NETWORK BASED END TO END MODEL FOR JOINT HEIGHT AND AGE ESTIMATION FROM SHORT DURATION SPEECH](https://ieeexplore.ieee.org/document/8683397)
5. [SHORT-TERM ANALYSIS FOR ESTIMATING PHYSICAL PARAMETERS OF SPEAKERS](http://mlsp.cs.cmu.edu/people/rsingh/docs/bodyparams_iwbf.pdf)
6. [Multitask Speaker Profiling for Estimating Age, Height, Weight and Smoking Habits from Spontaneous Telephone Speech Signals](https://www.researchgate.net/publication/267636866_Multitask_Speaker_Profiling_for_Estimating_Age_Height_Weight_and_Smoking_Habits_from_Spontaneous_Telephone_Speech_Signals)
7. [Exploring ANN Back-Ends for i-Vector Based Speaker Age Estimation](http://cs.joensuu.fi/pages/tkinnu/webpage/pdf/is2015_ageivector.pdf)
8. [End-to-End Deep Neural Network Age Estimation](https://danielpovey.com/files/2018_interspeech_age_estimation.pdf)
9. [Age Estimation from Telephone Speech using i-vectors](https://www.researchgate.net/publication/289626518_Age_estimation_from_telephone_speech_using_i-vectors)
10. [Age Estimation in Short Speech Utterances Based on LSTM Recurrent Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8316819)
11. [SPEAKER AGE ESTIMATION ON CONVERSATIONAL TELEPHONE SPEECH USING SENONE POSTERIOR BASED I-VECTORS](https://ieeexplore.ieee.org/document/7472637)
12. [SIMPLIFIED AND SUPERVISED I-VECTOR MODELING FOR SPEAKER AGE REGRESSION](https://www.researchgate.net/publication/269295042_Simplified_and_supervised_i-vector_modeling_for_speaker_age_regression)
13. [Automatic speaker profiling from short duration speech data](https://www.sciencedirect.com/science/article/abs/pii/S0167639319301074)
</br>

### Popular Datasets:

1. **[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1):**

    - No. of utterances: 6300
    - No. of Speaker: 630 and 8 Dialects
    - Sampling Rate: 16 kHz
    - Purpose: ASR & Transcription
    - Male : Female :: 2:1
    - No. of Samples/ person: 10
    - Includes time-aligned orthographic, phonetic and word transcriptions
    - License: Copyright 1993 Linguistic Data Consortium
    - Year: 1993 </br>
</br>

2. **[NIST SRE 2010](https://catalog.ldc.upenn.edu/LDC2017S06):**

    - No. of utterances: 5583
    - No. of Speakers: 442
    - Quality: Telephone
    - Contains 2,255 hours of American English telephone speech and speech recorded over a microphone channel involving an interview scenario
    - Sampling Rate: 8 kHz
    - Duration: ~ 5 mins
    - Year: 2017 </br>
 
 </br>
 
3. **[NIST SRE 2008](https://catalog.ldc.upenn.edu/LDC2011S08):**

    - No. of utterances: ~ 3500
    - No. of Speakers: ~ 350
    - Quality: Telephone
    - Contains 942 hours of multilingual telephone speech and English interview speech along with transcripts
    - Sampling Rate: 8 kHz.
    - Duration: ~ 3 mins
    - Year of publication: 2011 </br>
  </br>  
    
### Comparison of different Techniques/ Works:

|S. No. | Paper Cited           | Dataset Used          | Methodology                      | Gender  | RMSE   | MAE  | Correlation Coefficient |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------ | ---- | ----------------------- |
| 1.    | Kalluri et al. (2019) | TIMIT                 | GMM-Posteriors + DNN + SVR       | Male    | 7.60   | -    | -                       |
|       |                       |                       |                                  | Female  | 8.63   | -    | -                       |
| 2.    | Singh et al. (2016)   | TIMIT                 | GMM + Random Forest      | Male    | 8.10   | 5.70    | -                       |
|       |                       |                       |                                  | Female  | 9.10   | 6.20    | -                       |
| 3.    | Babu et al. (2020)    | TIMIT                 | Fstats + Formant + Harmonic Features + SVR       | Male    | 8.10   | 5.20   | -                       |
|       |                       |                       |                                  | Female  | 8.70   | 5.60    | -                       |
| 4.    | Poorjam et al. (2014) | NIST SRE              | i-Vectors + Linear Kernel SVR       | Male    | -   | -    | 0.76                      |
|       |                       |                       |                                  | Female  | -    | -    | 0.85                       |
| 5.    | Fedorovai et al. (2015)) | NIST SRE              | MFCC + i-Vectors + ANN       | Male    | -      | 6.42   | 0.75                       |
|       |                       |                       |                                  | Female  | -     | 5.56    | 0.81                      |
| 6.    | Ghahremani et al. (2018) | NIST SRE              | Fusion (i-Vector + x-Vector) + LDA    | Male    | -   | 5.84    | 0.83                     |
|       |                       |                       |                                  | Female  | -   | 4.68    | 0.92                     |
| 7.    | Bahari et al. (2012) | NIST SRE              | i-Vectors + SVR      | Male    | -      | 7.63     | -                     |
|       |                       |                       |                                  | Female  | -   | 7.61    | -                       |
| 8.    |Zazo et al. (2018) | NIST SRE              | MFCC + LSTMs      | Male    | -      | 7.29    | 0.48                    |
|       |                       |                       |                                  | Female  | -   | 6.97    | 0.65                       |
| 9.    |Sadjadi et al. (2016) | NIST SRE              |fMLLR + i-Vectors      | Male    | -      | 4.70    | 0.89                    |
|       |                       |                       |                                  | Female  | -   | 4.70    | 0.91                       |

</br>

### Resources & Codes:

I shall be updating this section as I go along with this project.

</br></br>

## Height:

### Publications:

1. [A DEEP NEURAL NETWORK BASED END TO END MODEL FOR JOINT HEIGHT AND AGE ESTIMATION FROM SHORT DURATION SPEECH](https://ieeexplore.ieee.org/document/8683397)
2. [SHORT-TERM ANALYSIS FOR ESTIMATING PHYSICAL PARAMETERS OF SPEAKERS](http://mlsp.cs.cmu.edu/people/rsingh/docs/bodyparams_iwbf.pdf)
3. [Multitask Speaker Profiling for Estimating Age, Height, Weight and Smoking Habits from Spontaneous Telephone Speech Signals](https://www.researchgate.net/publication/267636866_Multitask_Speaker_Profiling_for_Estimating_Age_Height_Weight_and_Smoking_Habits_from_Spontaneous_Telephone_Speech_Signals)
4. [Automatic speaker profiling from short duration speech data](https://www.sciencedirect.com/science/article/abs/pii/S0167639319301074)
5. [SPEAKER HEIGHT ESTIMATION COMBINING GMM AND LINEAR REGRESSION SUBSYSTEMS](https://ieeexplore.ieee.org/document/6639131)
6. [Height Estimation from Speech Signals using i-vectors and Least-Squares Support Vector Regression](https://www.researchgate.net/publication/305860549_Height_estimation_from_speech_signals_using_i-vectors_and_least-squares_support_vector_regression)
7. [Estimation of unknown speaker’s height from speech](https://link.springer.com/article/10.1007/s10772-010-9064-2)
8. [Estimating Speaker Height and Subglottal Resonances Using MFCCs and GMMs](http://www.seas.ucla.edu/spapl/paper/Arsikere_2014_SPL.pdf)
9. [Automatic estimation of the first three subglottal resonances from adults’ speech signals with application to speaker height estimation](https://www.sciencedirect.com/science/article/abs/pii/S0167639312000805)

</br>

### Popular Datasets:

1. **[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1):**

    - No. of utterances: 6300
    - No. of Speaker: 630 and 8 Dialects
    - Sampling Rate: 16 kHz
    - Purpose: ASR & Transcription
    - Male : Female :: 2:1
    - No. of Samples/ person: 10
    - Includes time-aligned orthographic, phonetic and word transcriptions
    - License: Copyright 1993 Linguistic Data Consortium
    - Year: 1993 </br>
</br>

2. **[NIST SRE 2010](https://catalog.ldc.upenn.edu/LDC2017S06):**

    - No. of utterances: 5583
    - No. of Speakers: 442
    - Quality: Telephone
    - Contains 2,255 hours of American English telephone speech and speech recorded over a microphone channel involving an interview scenario
    - Sampling Rate: 8 kHz
    - Duration: ~ 5 mins
    - Year: 2017 </br>
 
 </br>
 
3. **[NIST SRE 2008](https://catalog.ldc.upenn.edu/LDC2011S08):**

    - No. of utterances: ~ 3500
    - No. of Speakers: ~ 350
    - Quality: Telephone
    - Contains 942 hours of multilingual telephone speech and English interview speech along with transcripts
    - Sampling Rate: 8 kHz.
    - Duration: ~ 3 mins
    - Year of publication: 2011 </br>
    
</br>

### Comparison of different Techniques/ Works:

|S. No. | Paper Cited           | Dataset Used          | Methodology                      | Gender  | RMSE   | MAE  | Correlation Coefficient |
| ----- | --------------------- | --------------------- | -------------------------------- | ------- | ------ | ---- | ----------------------- |
| 1.    | Kalluri et al. (2019) | TIMIT                 | GMM-Posteriors + DNN + SVR       | Male    | 6.85   | -    | -                       |
|       |                       |                       |                                  | Female  | 6.29   | -    | -                       |
| 2.    | Singh et al. (2016)   | TIMIT                 | GMM + Random Forest      | Male    | 7.00   | 5.30    | -                       |
|       |                       |                       |                                  | Female  | 6.50   | 5.20    | -                       |
| 3.    | Babu et al. (2020)    | TIMIT                 | Fstats + Formant + Harmonic Features + SVR       | Male    | 6.80   | 5.20   | -                       |
|       |                       |                       |                                  | Female  | 6.10   | 4.80    | -                       |
| 4.    | Williams et al. (2013) | TIMIT                 | Fusion (MFTR + GMM-HDBC)       | Male    | -     | 5.37    | -                       |
|       |                       |                       |                                  | Female  | -     | 5.49    | -                       |
| 5.    | Poorjam et al. (2015)   | NIST SRE                 | i-Vector + LSSVR      | Male    | -     |  -     | 0.41                       |
|       |                       |                       |                                  | Female  | -     | -       | 0.40                       |
| 6.    | Mporas et al. (2009)    | TIMIT                 | MFCC + Bagging       | Male    | 6.80   | 5.30   | -                       |
|       |                       |                       |                                  | Female  | 6.40   | 5.20    | -                       |
| 7.    | Arsikere et al. (2013))    | TIMIT                 | MFCC + SGR + GMM       | Male    | 6.40   | -    | -                       |
|       |                       |                       |                                  | Female  | 5.80    | -     | -                       |

</br>

### Resources & Codes:
I shall be updating this section as I go along with this project.

</br></br>

## Accent:

### Publications:

1. [AN EMPIRICAL STUDY OF AUTOMATIC ACCENT CLASSIFICATION](https://groups.csail.mit.edu/sls/publications/2008/ICASSP08_Choueiter_MSR.pdf)
2. [Language accent classification in American English](http://ccc.inaoep.mx/~villasen/bib/languageaccent.pdf)
3. [VFNet: A Convolutional Architecture for Accent Classification](https://arxiv.org/pdf/1910.06697.pdf)
4. [The INTERSPEECH 2016 Computational Paralinguistics Challenge: Deception, Sincerity & Native Language](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0129.PDF)
5. [Speech Accent Classification](http://cs229.stanford.edu/proj2017/final-reports/5238301.pdf)
6. [Accent Identification by Combining Deep Neural Networks and Recurrent Neural Networks Trained on Long and Short Term Features](https://pdfs.semanticscholar.org/93e9/13b002e79436042af0689bb1ac2927a3180c.pdf)

</br>

### Popular Datasets:

1. **[CSLU Foreign-Accented English (FAE) Dataset](https://catalog.ldc.upenn.edu/LDC2007S08):**

  - No. of utterances: 4925
  - No. of Accents: 23
  - Quality: Telephone
  - Duration: ~20 sec
  - Type of Speakers: All Non-Native
  - Three native speakers of American English independently listened to each utterance and judged the speakers' accents on a 4-point scale:
    - negligible/no accent,
    - mild accent,
    - strong accent and
    - very strong accent.
  - Year: 2007
  
  </br>
  
2. **[TS Corpus of Non-Native Spoken English](https://catalog.ldc.upenn.edu/LDC2014T06):**

  - No. of utterances: 5132
  - No. of Accents: 11
  - Sampling Rate: 16 Hz
  - Type of Speakers: All Non-Native
  - Year: 2014
  
</br>

3. **[Speech Accent Archive](https://accent.gmu.edu/):**
  - No. of Samples: 2140
  - Type of Speakers: Native & non-native
  - Purpose: Speaker Profiling
  - Year: 2013
  - Speakers with 214 different native languages.
  - Speakers from 177 different countries
  - Questions answered by subjects:
    - Where were you born?
    - What is your native language?2
    - What other languages besides English and your native language do you know?
    - How old are you?
    - How old were you when you first began to study English?
    - How did you learn English? (academically or naturalistically)
    - How long have you lived in an english-speaking country? Which country?
  - License: CC BY-NC-SA 4.0

</br>

### Comparison of different Techniques/ Works:

|S. No. | Paper Cited           | Dataset Used          | Methodology                      | UAR   | Accuracy/ Detection Rate |
| ----- | --------------------- | --------------------- | -------------------------------- | ----- | ---------------------- |  
| 1.    | Schuller et al. (2016)| ETS Corpus            | 16-bit signed integer PCM WAV + SVM | 47.5% | - |
| 2.    | Choueiter et al. (2008)   | CLSU Foreign Accented English Corpus                 | GT + MMI + HLDA     | - | 32.7% |
| 3.    | Ahmed et al. (2019)    | Speech Accent Archive                 | Spectrograms + CNNs       | - | 70.33% |
| 4.    | Williams et al. (2013) | Speech Accent Archive                 | MFCC + LSTM     | - | 52.27% |
| 5.    | Poorjam et al. (2015)   | ETS Corpus                 | MFCC + DNN + RNN     | 50.40% | 50.20% |

### Resources & Codes:
I shall be updating this section as I go along with this project.

</br></br>
