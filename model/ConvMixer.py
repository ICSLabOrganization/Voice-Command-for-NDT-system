import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Permute, ReLU, Softmax, DepthwiseConv2D, SeparableConv2D, Input, Add, Multiply, Activation
from tensorflow.keras import layers
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2

EPOCHS = 100
LEARNING_RATE = 0.0001

# model architecture
channels = 1
columns = 10
rows = int(input_length / (columns * channels))

filters1 = 16
filters2 = filters1*2
filters3 = filters1*4
weight_decay = 1e-4
regularizer = l2(weight_decay)
final_pool_size = (int(rows/2), int(columns/2))

inputs = Input((None,input_length))

x = Reshape((rows, columns, channels), input_shape=(input_length, ))(inputs)

x = Conv2D(filters=filters1, kernel_size=3, strides=(2, 1), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

shortcut1 = Conv2D(filters1, kernel_size=3)(x)
shortcut1 = BatchNormalization()(shortcut1)
shortcut1 = Activation('relu')(shortcut1)
#frequency domain
f1 = Conv2D(filters = int(x.shape[-1]/4), kernel_size=1, strides=1, padding='same')(x)
f1 = BatchNormalization()(f1)
f1 = Activation('relu')(f1)
dw1 = DepthwiseConv2D(kernel_size=3)(f1)
dw1 = BatchNormalization()(dw1)
dw1 = Activation('relu')(dw1)
att1 = layers.GlobalAveragePooling2D()(dw1)
att1 = Dense((x.shape[-1]/4)/4, activation='relu')(att1)
att1 = Dense(x.shape[-1]/4, activation='sigmoid')(att1)
scaled1 = Multiply()([dw1, att1])
freq1 = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(scaled1)
#temporal domain
temp1 = DepthwiseConv2D(kernel_size=(freq1.shape[1], 3), strides=1, padding='same')(freq1)
temp1 = BatchNormalization()(temp1)
temp1 = Activation('relu')(temp1)
temp1 = Conv2D(filters=filters1, kernel_size=1, padding='same')(temp1)
temp1 = BatchNormalization()(temp1)
temp1 = Activation('relu')(temp1)
interact1 = Add()([freq1, temp1])
interact1 = Add()([shortcut1, interact1])

shortcut2 = Conv2D(filters2, kernel_size=3)(interact1)
shortcut2 = BatchNormalization()(shortcut2)
shortcut2 = Activation('relu')(shortcut2)
#frequency domain
f2 = Conv2D(filters = int(interact1.shape[-1]/4), kernel_size=1, strides=1, padding='same')(interact1)
f2 = BatchNormalization()(f2)
f2 = Activation('relu')(f2)
dw2 = DepthwiseConv2D(kernel_size=3)(f2)
dw2 = BatchNormalization()(dw2)
dw2 = Activation('relu')(dw2)
att2 = layers.GlobalAveragePooling2D()(dw2)
att2 = Dense((interact1.shape[-1]/4)/4, activation='relu')(att2)
att2 = Dense(interact1.shape[-1]/4, activation='sigmoid')(att2)
scaled2 = Multiply()([dw2, att2])
freq2 = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(scaled2)
#temporal domain
temp2 = DepthwiseConv2D(kernel_size=(freq2.shape[1], 3), strides=1, padding='same')(freq2)
temp2 = BatchNormalization()(temp2)
temp2 = Activation('relu')(temp2)
temp2 = Conv2D(filters=filters2, kernel_size=1, padding='same')(temp2)
temp2 = BatchNormalization()(temp2)
temp2 = Activation('relu')(temp2)
interact2 = Add()([freq2, temp2])
interact2 = Add()([shortcut2, interact2])

shortcut3 = Conv2D(filters3, kernel_size=3)(interact2)
shortcut3 = BatchNormalization()(shortcut3)
shortcut3 = Activation('relu')(shortcut3)
#frequency domain
f3 = Conv2D(filters = int(interact2.shape[-1]/4), kernel_size=1, strides=1, padding='same')(interact2)
f3 = BatchNormalization()(f3)
f3 = Activation('relu')(f3)
dw3 = DepthwiseConv2D(kernel_size=3)(f3)
dw3 = BatchNormalization()(dw3)
dw3 = Activation('relu')(dw3)
att3 = layers.GlobalAveragePooling2D()(dw3)
att3 = Dense((interact2.shape[-1]/4)/4, activation='relu')(att3)
att3 = Dense(interact2.shape[-1]/4, activation='sigmoid')(att3)
scaled3 = Multiply()([dw3, att3])
freq3 = Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(scaled3)
#temporal domain
temp3 = DepthwiseConv2D(kernel_size=(freq3.shape[1], 3), strides=1, padding='same')(freq3)
temp3 = BatchNormalization()(temp3)
temp3 = Activation('relu')(temp3)
temp3 = Conv2D(filters=filters3, kernel_size=1, padding='same')(temp3)
temp3 = BatchNormalization()(temp3)
temp3 = Activation('relu')(temp3)
interact3 = Add()([freq3, temp3])
interact3 = Add()([shortcut3, interact3])

y = Conv2D(12, kernel_size=3, strides=2, padding='same')(interact3)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

y = DepthwiseConv2D(kernel_size=3, padding='same')(y)
y = BatchNormalization()(y)
y = Activation('relu')(y)

output = layers.GlobalAveragePooling2D()(y)
outputs = Softmax()(output)

model = Model(inputs = inputs, outputs = outputs)
model.summary()

# this controls the learning rate
opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
