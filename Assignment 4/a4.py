from keras import Sequential
from keras import layers
from keras import optimizers
from keras import regularizers
import sys
from gensim.models import word2vec
import numpy as np

def save_into_list(f, data, pos_or_neg, target, feature_num):
    raw_data = []
    # save each line into list
    for line in f.readlines():
        raw_data.append(eval(line))
    # vectorize the data
    raw_data = word2vec.Word2Vec(raw_data, size=feature_num).wv.vectors
    # append the result into datalist and targetlist
    for vec in raw_data:
        data.append(vec)
        target.append(pos_or_neg)

def classify(feature_num, hidden_layer, train_wv_data, train_target, val_wv_data, val_target ):
    # start building a neural network classifier
    model = Sequential()
    # Input layer of 
    model.add(layers.Dense(units = 30, input_dim=feature_num, activation='linear'))
    # One hidden layer. For the hidden layer, try the following activation functions: ReLU, sigmoid and tanh
    model.add(layers.Dense(units = 30, activation = hidden_layer))
    # Add L2-norm regularization
    model.add(layers.Dense(units = 30, kernel_regularizer=regularizers.l2(0.01)))
    # Add dropout. Try a few different dropout rates.
    model.add(layers.Dropout(0.3))
   
    # Final layer with softmax activation function.
    model.add(layers.Dense(units = 1, activation = 'softmax'))

    # Use cross-entropy as the loss function
    adam = optimizers.Adam()
    model.compile(loss='binary_crossentropy',optimizer =adam,  metrics=['accuracy'])

    # train data
    model.fit(np.array(train_wv_data), np.array(train_target))
    print("Training finished \n")
    # validation
    evalu = model.evaluate(np.array(val_wv_data), np.array(val_target), verbose=0)
    print("Evaluation on validation data: loss = %0.6f accuracy = %0.2f%% \n" % (evalu[0], evalu[1] * 100))


if __name__ == "__main__":

    train_wv_data = []
    train_target = []
    val_wv_data = []
    val_target = []
    test_wv_data = []
    test_target = []

    feature_num = 100
    # vectorize data and save data into list _wv_data, and save neg/pos into list _target
    for i in range(1,len(sys.argv)):
        f = open(sys.argv[i],'r') 
        file_name = sys.argv[i]
        pos_or_neg = 1 if 'pos' in file_name else 0

        if( "train" in file_name):
            save_into_list(f,train_wv_data, pos_or_neg, train_target, feature_num)
        elif( "test" in file_name):
            save_into_list(f,test_wv_data, pos_or_neg, test_target,feature_num)
        else:
            save_into_list(f,val_wv_data, pos_or_neg, val_target, feature_num)
        f.close()

    print("vectorize finished\n")

    hidden_layer = ['relu']

    #build the model with train_data and validate the model with val_data
    for hl in hidden_layer:
        print("The hidden layer is " + hl+"\n")
        classify(feature_num, hl, train_wv_data, train_target, val_wv_data, val_target )

