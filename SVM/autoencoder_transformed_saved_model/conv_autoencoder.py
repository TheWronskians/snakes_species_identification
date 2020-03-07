import numpy as np
from keras.datasets import mnist
from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pickle
import PIL

if __name__ == "__main__":

    trdata = ImageDataGenerator(rescale=1./255)

    traindata = trdata.flow_from_directory(directory="datasets/train",target_size=(224,224),color_mode='rgb',batch_size = 32,class_mode='input')
    # trdata.fit(traindata)

    tsdata = ImageDataGenerator(rescale=1./255)

    testdata = tsdata.flow_from_directory(directory="datasets/val", target_size=(224,224),color_mode='rgb',batch_size = 32,class_mode='input')

    input_img = Input(shape=(224,224,3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # At this point the representation is (7, 7, 32)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])

    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=1, mode='auto')


    checkpoint = ModelCheckpoint("autoencoder_1.h5", monitor='val_loss', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)

    hist = autoencoder.fit_generator(generator= traindata, steps_per_epoch= 2060, epochs= 25,
                          validation_data= testdata, validation_steps = 516,callbacks=[checkpoint,early])

    with open('autoencoder', 'wb') as handle:
        pickle.dump(hist.history, handle)

    # autoencoder.save_weights("autoencoder_1.h5")
