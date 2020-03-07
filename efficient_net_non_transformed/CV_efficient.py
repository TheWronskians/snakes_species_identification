from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import efficientnet.keras as efn

from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping

from efficientnet.keras import center_crop_and_resize, preprocess_input
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
import pickle



if __name__ == "__main__":
    efficient_net = efn.EfficientNetB3(weights='imagenet',include_top = False)

    trdata = ImageDataGenerator(rescale=1./255,processing_function = preprocess_input)

    traindata = trdata.flow_from_directory(directory="datasets/train",target_size=(224,224),color_mode='rgb',batch_size = 20)

    tsdata = ImageDataGenerator(rescale=1./255)

    testdata = tsdata.flow_from_directory(directory="datasets/val", target_size=(224,224),color_mode='rgb',batch_size = 20)


    x = efficient_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    num_class = np.unique(testdata.classes).shape[0]
    predictions = Dense(num_class, activation= 'softmax')(x)
    model = Model(inputs = efficient_net.input, outputs = predictions)

    model.trainable = True
    model.summary()

    num_val_steps = len(testdata.filenames)/20
    num_train_steps = len(traindata.filenames)/20

    model.compile(loss = "categorical_crossentropy",
                    optimizer = 'Adam', metrics=["accuracy"])

    checkpoint = ModelCheckpoint("efficientnetmodel.h5", monitor='val_acc', verbose=1,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')


    hist = model.fit_generator(generator= traindata, steps_per_epoch= num_train_steps, epochs= 25,
                              validation_data= testdata, validation_steps=num_val_steps, callbacks=[checkpoint,early])

    model.save_weights("efficient_net_weight.h5")

    with open('efficientnetHistory', 'wb') as handle:
        pickle.dump(hist.history, handle)
