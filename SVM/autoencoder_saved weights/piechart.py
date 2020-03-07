import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    tsdata = ImageDataGenerator(rescale=1./255)

    trdata = ImageDataGenerator(rescale=1./255)

    traindata = tsdata.flow_from_directory(directory="../datasets/train", target_size=(224,224),color_mode='rgb',batch_size = 32,class_mode='input',shuffle= False)

    testdata = tsdata.flow_from_directory(directory="../datasets/val", target_size=(224,224),color_mode='rgb',batch_size = 32,class_mode='input',shuffle= False)


    
