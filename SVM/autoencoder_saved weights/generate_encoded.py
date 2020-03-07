import numpy as np
from keras.datasets import mnist
from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model,Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import pickle
import PIL
from keras.models import load_model

if __name__ == "__main__":
    tsdata = ImageDataGenerator(rescale=1./255)

    trdata = ImageDataGenerator(rescale=1./255)

    traindata = tsdata.flow_from_directory(directory="../datasets/train", target_size=(224,224),color_mode='rgb',batch_size = 32,class_mode='input',shuffle= False)

    testdata = tsdata.flow_from_directory(directory="../datasets/val", target_size=(224,224),color_mode='rgb',batch_size = 32,class_mode='input',shuffle= False)

    # input_img = Input(shape=(224,224,3))
    #
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    #
    # x = MaxPooling2D((2, 2), padding='same')(x)
    #
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #
    # encoded = MaxPooling2D((2, 2), padding='same')(x)
    #
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    #
    # x = UpSampling2D((2, 2))(x)
    #
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #
    # x = UpSampling2D((2, 2))(x)
    #
    # decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    #
    # autoencoder = Model(input_img, decoded)
    #
    # autoencoder.summary()


    # autoencoder.load_weights("autoencoder_1.h5")
    #
    # nb_samples = len(testdata.filenames)/32
    #
    # autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])
    #
    # encoding_model = Model(inputs = [input_img],outputs = [encoded])
    # encoding_model.summary()
    #
    #
    # encoding_model_flat = Sequential()
    # encoding_model_flat.add(encoding_model)
    # encoding_model_flat.add(MaxPooling2D((4,4), padding='same'))
    # encoding_model_flat.add(Flatten())
    #
    # encoding_model_flat.summary()
    #
    #
    # train_sample_nb = len(traindata.filenames)/32
    #
    # test_sample_nb = len(testdata.filenames)/32
    #
    # encoding_model_flat.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])
    #
    # encoding_model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mean_squared_error"])
    #
    # generate_embedding_train = encoding_model_flat.predict(traindata,steps = train_sample_nb,verbose=1)
    #
    # generate_embedding_test = encoding_model_flat.predict(testdata,steps = test_sample_nb,verbose=1)
    #
    # with open('autoencoder_embedding_test', 'wb') as handle:
    #     pickle.dump(generate_embedding_test, handle)
    #
    # with open('autoencoder_embedding_train', 'wb') as handle:
    #     pickle.dump(generate_embedding_train, handle)
    #
    # with open('y_train_classes', 'wb') as handle:
    #     pickle.dump(traindata.classes, handle)
    #
    # print("hi")
    # print(generate_embedding_test.shape)
    #
    # print(generate_embedding_train.shape)
