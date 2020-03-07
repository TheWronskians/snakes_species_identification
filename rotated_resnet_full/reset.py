from keras.applications.resnet_v2 import ResNet101V2
import keras
from keras.models import Model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet_v2 import preprocess_input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
import pickle
import PIL


if __name__ == "__main__":
    resnet_model = ResNet101V2(weights='imagenet', include_top=False)

    trdata = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)

    traindata = trdata.flow_from_directory(directory="datasets/train",target_size=(224,224),color_mode='rgb',batch_size = 20)
    # trdata.fit(traindata)

    tsdata = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)

    testdata = tsdata.flow_from_directory(directory="datasets/val", target_size=(224,224),color_mode='rgb',batch_size = 20)
    x = resnet_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    num_class = np.unique(testdata.classes).shape[0]
    predictions = Dense(num_class, activation= 'softmax')(x)
    model = Model(inputs = resnet_model.input, outputs = predictions)

    # model.summary()
    for layer in model.layers[:]:
        layer.trainable=True


    model.compile(loss = "categorical_crossentropy",
                    optimizer = 'Adam', metrics=["accuracy"])

    checkpoint = ModelCheckpoint("resnet_1.h5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=40, verbose=1, mode='auto')


    hist = model.fit_generator(generator= traindata, steps_per_epoch= 3296, epochs= 25,
                              validation_data= testdata, validation_steps=0, callbacks=[checkpoint,early])

    with open('ResnetHistory', 'wb') as handle:
        pickle.dump(hist.history, handle)
    
