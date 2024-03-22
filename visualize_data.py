#%%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import librosa
import tensorflow as tf

#%%
classes = ["stop", "forward", "backward", 
           "left", "right", "up", "down",
           "_background_noise_"]

#%%
voiceDF = pd.read_csv('dataset/voicecommand_dataset.csv', header = None)
voice_commands = np.array(voiceDF)
voice_signal = voice_commands[:, :-1]
commands = voice_commands[:, -1].astype(np.int32)

# %%
num_classes = 8

sample_per_class = {}

for class_idx in range(num_classes):
    idx = np.where(commands == class_idx)[0][0]
    sample_per_class[class_idx] = voice_signal[idx]

row = 2
column = int(np.ceil(num_classes / row))
fig, axs = plt.subplots(row, column, figsize=(12, 5))

for i, class_idx in enumerate(sample_per_class):
    axs[i // column, i % column].plot(sample_per_class[class_idx])
    axs[i // column, i % column].set_title(classes[class_idx], fontsize=10)

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(17, 7))
for i in range(len(sample_per_class)):
    audio_signal = sample_per_class[i]
    
    sample_rate = 16000
    window_size_ms = 30
    dct_coefficient_count = 10
    window_stride_ms = 20
    clip_duration_ms = 1000
    
    desired_samples = int(sample_rate * clip_duration_ms / 1000)

    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
      spectrogram_length = 0
    else:
      spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
      fingerprint_size = dct_coefficient_count * spectrogram_length
    
    stfts = tf.signal.stft(audio_signal, frame_length=window_size_samples, 
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
    print(mfccs.shape)
    plt.subplot(2, 4, i+1)
    plt.imshow(mfccs)
    plt.title(classes[i], fontsize=10)
plt.tight_layout()
plt.show()
# %%
