from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np
import argparse
import os
from PIL import Image
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import argparse
import imageio as io
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


def readImage(imagepath):
    im =  None
    try:
        image = Image.open(imagepath)
        image.copy().verify()
        im = np.asarray(image)
    except (IOError,SyntaxError) as e:
        print('Bad file:', imagepath)
    return im


def get_images_info(imageDir):
    all_imagesid = []
    all_image_paths = []
    count = 0
    for name in os.listdir(imageDir):
        curr_id_images = []
        for curr_image in os.listdir(imageDir + "/" + name):
            curr_path = imageDir + "/" + name + "/" + curr_image
            curr_id_images.append(curr_path)
        for p_pos in range(0,len(curr_id_images)):
            all_imagesid.append(name)
            all_image_paths.append(curr_id_images[p_pos])
    return all_image_paths,all_imagesid

def get_images_matrix(all_imagesdir,batchsize,startpoint):
    count = 0
    data_matrix = []
    for i in range(batchsize):
        new_image = readImage(all_imagesdir[startpoint])
        if new_image is not None:
            new_image = np.ndarray.flatten(np.array(Image.fromarray(new_image).resize((200,200),Image.BILINEAR))/255.)
            if count == 0:
                data_matrix = new_image
                count += 1
            else:
                data_matrix = np.vstack((data_matrix,new_image))
        startpoint +=1


    return data_matrix,startpoint


def main():
    imagesDir = 'round1_test'
    all_image_paths,all_imagesid = get_images_info(imagesDir)
    tf.reset_default_graph()
    num_inputs=200*200*3    #80x80x3 pixels
    num_hid1=512
    num_output=num_inputs
    lr=0.0033

    activation_func=tf.nn.relu

    X=tf.placeholder(tf.float32,shape=[None,num_inputs])
    initializer=tf.variance_scaling_initializer()

    w1=tf.Variable(initializer([num_inputs,num_hid1]),dtype=tf.float32)
    w2=tf.Variable(initializer([num_hid1,num_output]),dtype=tf.float32)
    b1=tf.Variable(tf.zeros(num_hid1))
    b2=tf.Variable(tf.zeros(num_output))
    hid_layer1= activation_func(tf.matmul(X,w1)+b1)
    output_layer= activation_func(tf.matmul(hid_layer1,w2)+b2)


    loss=tf.reduce_mean(tf.square(output_layer-X))
    optimizer=tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.9, epsilon=1.0)
    train=optimizer.minimize(loss)
    init=tf.global_variables_initializer()


    num_epoch=1
    batch_size=150
    num_test_images=10
    startpoint = 0

    # test_imagesDir = 'datasets/all_faces_data_randomized'
    # all_test_paths,all_test_id = get_images_info(test_imagesDir)
    # #test_data_mat = get_test_images_matrix(all_test_paths,10)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            num_batches= len(all_image_paths)//batch_size
            print("Epoch: %d/%d" %(epoch+1,num_epoch))
            np.random.shuffle(all_image_paths)
            for iteration in range(num_batches):
                X_batch,startpoint = get_images_matrix(all_image_paths,batch_size,startpoint)
                print("Current image list check point: %d" %(startpoint),end = '\r')
                sess.run(train,feed_dict={X:X_batch})
                print("Batch %d/%d "%(iteration+1,num_batches), end = '\r')
            startpoint = 0
            print("")
            train_loss=loss.eval(feed_dict={X:X_batch})
            print("epoch {} loss {}".format(epoch+1,train_loss))
            if epoch % 5 == 0 :
                saver.save(sess, "autoencoder_model/model-%d.ckpt" %(epoch))
            with open(('autoencoder_model/auto_encoder_mse.txt'),'at') as f:
                f.write('%2.5f\n'%(train_loss))


    sess.close()




if __name__ == "__main__":
    main();
