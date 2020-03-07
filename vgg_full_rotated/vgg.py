from keras.applications.vgg16 import VGG16
import keras
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pickle import dump


if __name__ == "__main__":



    vggmodel = VGG16(weights='imagenet', include_top=False)

    trdata = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)

    traindata = trdata.flow_from_directory(directory="datasets/train",color_mode='rgb',batch_size= 32)
    # trdata.fit(traindata)

    tsdata = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)

    testdata = tsdata.flow_from_directory(directory="datasets/val",color_mode='rgb',batch_size= 32)


    num_class = np.unique(testdata.classes).shape[0]


    x=vggmodel.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3

    preds = Dense(num_class, activation="softmax")(x)
    model_final = Model(inputs = vggmodel.input, outputs = preds)


    for layer in model_final.layers[:19]:
        layer.trainable=False
    for layer in model_final.layers[19:]:
        layer.trainable=True


    model_final.compile(loss = "categorical_crossentropy",
                        optimizer = optimizers.adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                        metrics=["accuracy"])


    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=40, verbose=1, mode='auto')


    hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 2060, epochs= 30,
                              validation_data= testdata, validation_steps=0, callbacks=[checkpoint,early])


    with open('History', 'wb') as handle: # saving the history of the model
        dump(hist.history, handle)
