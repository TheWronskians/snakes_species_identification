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
from keras.models import load_model
import pickle

if __name__ == "__main__":

    vggmodel = VGG16(weights='imagenet', include_top=False)

    x=vggmodel.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3

    tsdata = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)
    testdata = tsdata.flow_from_directory(directory="../datasets/val",color_mode='rgb',batch_size= 100,shuffle = False)
    filenames = testdata.filenames

    nb_samples = len(filenames)/100

    num_class = np.unique(testdata.classes).shape[0]
    preds = Dense(num_class, activation="softmax")(x)
    myModel = Model(inputs = vggmodel.input, outputs = preds)



    myModel = load_model("resnet_1.h5")

    myModel.compile(loss = "categorical_crossentropy",
                        optimizer = 'adam', metrics=["accuracy"])

    predictions = myModel.predict_generator(testdata,steps = nb_samples,verbose=1)

    y_pred_classes = predictions.argmax(axis = -1)

    with open('y_predict', 'wb') as handle:
        pickle.dump(y_pred_classes, handle)
    
    print(y_pred_classes)
    print(testdata.classes)
