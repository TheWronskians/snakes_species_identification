from sklearn.svm import SVC
import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":
    model = SVC(kernel='sigmoid', probability=False,max_iter = 20000,random_state = 42,verbose = 1)

    print("Loading training and test data")
    pickle_in = open("autoencoder_embedding_train","rb")
    train_data = pickle.load(pickle_in)
    pickle_in = open("y_train_classes","rb")
    train_label = pickle.load(pickle_in)

    pickle_in = open("autoencoder_embedding_test","rb")
    test_data = pickle.load(pickle_in)
    pickle_in = open("y_real_class","rb")
    test_label = pickle.load(pickle_in)

    samplesize = 50

    print("Choosing subset of train data for training and testing.")
    sub_train_set = []
    sub_train_label = []
    count = 0
    for i in range(44):
        pos = np.where(train_label == i)[0]
        sample_pos = np.random.choice(pos,samplesize,replace=False)
        if count == 0:
            sub_train_set = train_data[sample_pos]
            sub_train_label = train_label[sample_pos]
            count += 1
        else:
            sub_train_set = np.concatenate([sub_train_set,train_data[sample_pos]])
            sub_train_label = np.concatenate([sub_train_label,train_label[sample_pos]])


    scaling = MinMaxScaler(feature_range=(-1,1)).fit(sub_train_set)
    sub_train_set = scaling.transform(sub_train_set)
    test_data = scaling.transform(test_data)



    print("Fitting model")

    model.fit(sub_train_set,sub_train_label)


    print("Done!")
    print("Getting test accuracy")
    acc = model.score(test_data,test_label)
    print(acc)
