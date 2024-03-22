import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, InputLayer, Dropout, Conv1D, Conv2D, Flatten, Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D, BatchNormalization, Permute, ReLU, Softmax, Input, Activation, concatenate
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2

EPOCHS = 100
LEARNING_RATE = 0.0001

DenseNetB = True
DenseNetC = True
NB_BLOCKS = 3
NB_DEPTH = 4
GROWTH_RATE = 3
COMPRESSION_FACTOR = 0.5

channels = 1
columns = 10
rows = 49

def composite_function(x, growth_rate):
    if DenseNetB: #Add 1*1 convolution when using DenseNet B
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(kernel_size=(1,1), strides=1, filters = 4 * growth_rate, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    output = Conv2D(kernel_size=(3,3), strides=1, filters = growth_rate, padding='same')(x)
    return output

def dense_block(x, depth=5, growth_rate = 3):
    nb_input_feature_map = x.shape[3]
    stored_features = x
    for i in range(depth):
        feature = composite_function(stored_features, growth_rate = growth_rate)
        stored_features = concatenate([stored_features, feature], axis=3)
    return stored_features

def dense_net(inputs, nb_blocks = 2):
    x = Reshape((rows, columns, channels), input_shape=(input_length, ))(inputs)
    x = Conv2D(kernel_size=(3,3), filters=8, strides=1, padding='same', activation='relu')(x)
    for block in range(nb_blocks):
        x = dense_block(x, depth=NB_DEPTH, growth_rate = GROWTH_RATE)
        if not block == nb_blocks-1:
            if DenseNetC:
                theta = COMPRESSION_FACTOR
            nb_transition_filter =  int(x.shape[3] * theta)
            x = Conv2D(kernel_size=(1,1), filters=nb_transition_filter, strides=1, padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2,2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(12, activation='softmax', kernel_initializer='he_normal')(x)
    return outputs

input_image = Input((None, input_length))
model = Model(input_image, dense_net(input_image, NB_BLOCKS))
model.summary()

# this controls the learning rate
opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)
callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS, ensure_determinism=ENSURE_DETERMINISM))

# train the neural network
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset, verbose=2, callbacks=callbacks)
