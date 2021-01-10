# SPEC AUGMENT (without axis-wrapping)

import random

def spec_augment(mel_spectrogram, time_warping_para=80, frequency_masking_para=27,
                 time_masking_para=100, frequency_mask_num=2, time_mask_num=2):
    
    v = mel_spectrogram.shape[1]
    tau = mel_spectrogram.shape[0]
    mel_spectrogram_copy = mel_spectrogram.copy()
    
    # Step 1 : Frequency masking
    for i in range(frequency_mask_num):
        f0 = random.randint(0, v-3)
        f=3
        mel_spectrogram_copy[:, f0:f0 + f] = 0

    # Step 2 : Time masking
    for i in range(time_mask_num):
        t = 30
        t0 = random.randint(0, tau - 30)
        mel_spectrogram_copy[t0:t0 + t, :] = 0

    return mel_spectrogram_copy