#%%
import numpy as np 
import tensorflow as tf
from tqdm import tqdm

# %%
classes = ["stop", "forward", "backward", 
           "left", "right", "up", "down",
           "_background_noise_"]


#%%
def convert_scale(input_audio, sample_rate, 
                  window_size_ms, dct_coefficient_count, 
                  window_stride_ms, clip_duration_ms): 
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
      spectrogram_length = 0
    else:
      spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = dct_coefficient_count * spectrogram_length
    
    stfts = tf.signal.stft(input_audio, frame_length=window_size_samples, 
                         frame_step=window_stride_samples, fft_length=None,
                         window_fn=tf.signal.hann_window
                         )
    spectrograms = tf.abs(stfts)
    spectrograms = tf.cast(spectrograms, tf.float32)
    
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40 
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                        sample_rate,
                                                                        lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :dct_coefficient_count]
    mfccs = tf.reshape(mfccs,[spectrogram_length, dct_coefficient_count, 1])
    return mfccs
  
#%%
import pandas as pd
df = pd.read_csv('extractedset.csv')
data = np.array(df)
a = convert_scale(data[0], 16000, 30, 10, 20, 1000)

# %%
def load_data(voiceDF, load_mfcc=False, 
              sample_rate=16000, window_size=30, 
              dct_coefficient_count=10, window_stride_ms=20, 
              clip_duration_ms=1000):
    print("Loading Data...")
    voice_commands = np.array(voiceDF)
    voice_signal = voice_commands[:, :-1]
    commands = voice_commands[:, -1].astype(np.int32)
    
    voice_data = []
    labels = []
    for i in tqdm(range(len(voice_signal))):
        audio = voice_signal[i]
        if load_mfcc == True:
            mfcc = convert_scale(audio, sample_rate=sample_rate, 
                                      window_size_ms=window_size, 
                                      dct_coefficient_count=dct_coefficient_count,
                                      window_stride_ms=window_stride_ms,
                                      clip_duration_ms=clip_duration_ms)
            voice_data.append(mfcc)
        else:
            voice_data.append(audio)
        labels.append(commands[i])
    voice_data = np.array(voice_data)
    labels = np.array(labels).astype(np.int32)
    
    print("Data loaded!")
    print("Data shape: ",voice_data.shape)    
    print("Label shape: ", labels.shape)
    return voice_data, labels
