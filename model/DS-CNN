import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv2D, Flatten, Reshape, MaxPooling2D, AveragePooling2D, BatchNormalization, Permute, ReLU, Softmax, DepthwiseConv2D, SeparableConv2D, Input
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2

EPOCHS = 200
LEARNING_RATE = 0.0001
BATCH_SIZE = args.batch_size or 32

# model size info
# S: 5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1
# M: 172 10 4 2 1 172 3 3 2 2 172 3 3 1 1 172 3 3 1 1 172 3 3 1 1
# L: 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1

# model architecture
channels = 1
columns = 10
rows = int(input_length / (columns * channels))

inputs = Input((None,input_length))

x = Reshape((rows, columns, channels), input_shape=(input_length, ))(inputs)

x = Conv2D(172, (10,4), (2,1), padding='same', kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = SeparableConv2D(172, (3,3), (2,2), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = SeparableConv2D(172, (3,3), (1,1), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = SeparableConv2D(172, (3,3), (1,1), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = SeparableConv2D(172, (3,3), (1,1), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Dropout(0.2)(x)

x = AveragePooling2D(pool_size = (int(rows/4), int(columns/2)))(x)
x = Flatten()(x)

outputs = Dense(classes, activation='softmax')(x)

model = Model(inputs = inputs, outputs = outputs)
model.summary()

# this controls the learning rate
opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
