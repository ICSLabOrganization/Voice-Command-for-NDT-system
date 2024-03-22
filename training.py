#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import l2
from load_data import load_data
import os
from sklearn.metrics import confusion_matrix
# from main_layers import ConvMixerBlock

# %%
voiceDF = pd.read_csv('dataset/googlecommandV2_12.csv', header = None)
data, labels = load_data(voiceDF, load_mfcc=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# X_train, X_test = np.expand_dims(X_train, axis=-1), np.expand_dims(X_test, axis=-1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

classes_name = ["yes", "no", "up", "down",
                "left", "right", "on", "off",
                "silence", "stop", "go", "unknown"]
print(classes_name)
classes = np.unique(labels).astype(np.int32)
num_classes = len(classes)

#%%
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have defined num_classes, labels, data, and classes_name appropriately

sample_per_class = {}

for class_idx in range(num_classes):
    idx = np.where(labels == class_idx)[0][0]
    sample_per_class[class_idx] = data[idx]

row = 2
column = int(np.ceil(num_classes / row))
fig, axs = plt.subplots(row, column, figsize=(12, 5))

for i, class_idx in enumerate(sample_per_class):
    axs[i // column, i % column].imshow(sample_per_class[class_idx], aspect='auto', cmap='coolwarm')  # Adjusted aspect ratio
    axs[i // column, i % column].set_title(classes_name[class_idx], fontsize=10)
plt.suptitle("MFCC of each class")
plt.tight_layout()
plt.show()


#%%
train_class_counts = np.bincount(y_train.astype(np.int32))
test_class_counts = np.bincount(y_test.astype(np.int32))

num_classes = max(len(train_class_counts), len(test_class_counts))

# Set the width of the bars
bar_width = 0.35

# Define the x-axis positions for the bars
x_train = np.arange(num_classes)
x_test = [x + bar_width for x in x_train]

# Plotting the grouped bar plot
plt.figure(figsize=(10, 6))

plt.bar(x_train, train_class_counts, width=bar_width, label='Training')
plt.bar(x_test, test_class_counts, width=bar_width, label='Testing')

plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Samples per Class in Training and Testing Sets', fontsize=11)
plt.xticks([x + bar_width / 2 for x in range(num_classes)], range(num_classes))
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA for training data
pca_train = PCA(n_components=2)
X_train_pca = pca_train.fit_transform(X_train.reshape(X_train.shape[0], 490))

# PCA for testing data
pca_test = PCA(n_components=2)
X_test_pca = pca_test.fit_transform(X_test.reshape(X_test.shape[0], 490))

# Plotting the figures
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Training Data')
plt.colorbar(label='Class')
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of Testing Data')
plt.colorbar(label='Class')
plt.grid()

plt.tight_layout()
plt.show()

# %%
input_shape = X_train.shape[1:]
filters1 = 16
filters2 = filters1*2
filters3 = filters1*4
weight_decay = 1e-4
regularizer = l2(weight_decay)
final_pool_size = (int(input_shape[0]/2), int(input_shape[1]/2))
rows = X_train[0].shape[0]
columns = X_train[1].shape[1]

# %%
inputs = Input(shape=X_train[0].shape)
x = layers.Reshape((X_train[0].shape[0], X_train[0].shape[1], 1))(inputs)

x = layers.Conv2D(16, (10,4), strides=(2,2), padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

#1
x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
    
x = layers.Conv2D(32, kernel_size=1, padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)

#2
x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
    
x = layers.Conv2D(64, kernel_size=1, padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)


#3
x = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
    
x = layers.Conv2D(128, kernel_size=1, padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.AveragePooling2D(pool_size = (int(rows/2), int(columns/2)))(x)
x = layers.Conv2D(12, kernel_size=1, padding='same', kernel_regularizer=l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
output = layers.Activation('softmax')(x)
model = tf.keras.Model(inputs, output)
model.summary()

# %%
opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = ['accuracy']
model.compile(opt, loss=loss, metrics=metrics)
checkpoint_filepath = 'trained_models/model_pj'
def get_callbacks(patience = 5):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1,
                                                     patience=patience//2,
                                                     min_lr=1e-12,
                                                     verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                    save_weights_only=False,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True,
                                                    verbose=1)
    return [reduce_lr, checkpoint]

callbacks = get_callbacks(patience = 10)
# %%
history = model.fit(X_train, y_train,
                    batch_size = 128, epochs = 50,
                    validation_split=0.2,
                    callbacks = callbacks,
                    verbose = 1)

# %%
trained_model = tf.keras.models.load_model(checkpoint_filepath)
trained_model.evaluate(X_test, y_test)

# %%
import seaborn as sn
def plot_confusion_matrix(true_label, predict_label, pl = False):

    cm = confusion_matrix(y_true = true_label, y_pred = predict_label)
    cm_per = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]*100,2)
    if pl == True:
        df_cm = pd.DataFrame(cm_per, index = classes_name, columns = classes_name)
        plt.figure(figsize = (9, 8))
        sn.heatmap(df_cm, annot=True, cmap = "Reds", linewidths=.1 ,fmt='.0f')
        plt.title("Confusion matrix")
        plt.xlabel('Predict', fontsize=8)
        plt.ylabel('True', fontsize=8)
        plt.tight_layout()
        
# %%
pred = trained_model.predict(X_test)
y_pred = np.argmax(pred, axis=1)
prob_labels = np.max(pred, axis=1)
plot_confusion_matrix(y_test, y_pred, pl=True)

# %%
losses = history.history["loss"]
val_losses = history.history["val_loss"]
lr = history.history["lr"]

# %%
plt.figure(figsize=(9, 3.5))
plt.subplot(1, 2, 1)
plt.plot(lr)
plt.grid()
plt.xlabel("epoch")
plt.ylabel("learning rate")
plt.title("Learning rate schedule")

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.plot(val_losses)
plt.legend(["train", "val"])
plt.grid()
plt.title("Learning process")
plt.xlabel("epoch")
plt.ylabel("loss")

# %%
