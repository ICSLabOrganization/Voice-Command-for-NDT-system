#%%
import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.io import wavfile
from tqdm import tqdm

# %%

root_path = 'dataset/speech_command_dataset'

# %%
classes = ["_background_noise_", "go", "yes", 
           "no", "up", "down", "left", "right", 
           "on", "off", "stop", "unknown"]

unknown_words = ["bed", "bird", "cat", "dog", "happy", 
                 "house", "marvin", "sheila", "tree", "wow"]

# %%
data = []
labels = []
for i in tqdm(range(len(classes))):
    if classes[i] != "_background_noise_" and classes[i] != "unknown":
        print("Loading target classes...")
        files_name = os.listdir(os.path.join(root_path, classes[i]))
        print(classes[i].upper(), "includes", len(files_name), "sample(s)")
        for file_name in files_name:
            file_path = os.path.join(root_path, classes[i], file_name)
            _, datum = wavfile.read(file_path)
            if len(datum) < 16000:
                padding = np.zeros(16000 - len(datum))
                datum = np.concatenate((datum, padding))
            data.append(datum)
            labels.append(i) 
            
    if classes[i] == "_background_noise_":
        print("\nLoading background noise...")
        files_name = os.listdir(os.path.join(root_path, classes[i]))
        print(classes[i].upper(), "includes", len(files_name), "sample(s)")
        count_sample = 0
        for file_name in files_name:
            file_path = os.path.join(root_path, classes[i], file_name)
            _, datum2 = wavfile.read(file_path)
            sample_break = np.zeros(16000)
            count = 0
            
            for j in tqdm(range(len(datum2))):
                sample_break[count] = datum2[j]
                count += 1
                
                if count == 16000:
                    data.append(sample_break.copy())
                    labels.append(i) 
                    count_sample += 1
                    count = 0
                    sample_break.fill(0)
                    
                
        print(f"Loaded {count_sample} background_noise sample(s)")
            
    if classes[i] == "unknown":
        print("Loading unknown words...")
        for k in tqdm(range(len(unknown_words))):
            files_name = os.listdir(os.path.join(root_path, unknown_words[k]))
            print(classes[i].upper(), "includes", len(files_name), "samples(s)")
            for file_name in files_name:
                file_path = os.path.join(root_path, unknown_words[k], file_name)
                _, datum = wavfile.read(file_path)
                
                if len(datum) < 16000:
                    padding = np.zeros(16000 - len(datum))
                    datum = np.concatenate((datum, padding))
                data.append(datum)
                labels.append(i) 
        
data = np.array(data)
labels = np.array(labels)
print(data.shape)
print(labels.shape)

# %%
labels_ = np.expand_dims(labels, axis=1)
data_to_save = np.concatenate((data, labels_), axis=1)
np.savetxt('dataset/googlecommandV2_12.csv', data_to_save, delimiter=',')

# %%
