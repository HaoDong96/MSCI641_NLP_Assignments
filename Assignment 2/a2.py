import csv
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

def save_into_list(f, data, pos_or_neg, target):
    # save each line into list
    for line in f.readlines():
        line_str = ""
        line = line[1:-2].split(r", ")
        for i in range(len(line)):
            line[i] = line[i][1:-1]
            line_str += str(line[i]+" ")
        data.append(line_str)
        target.append(pos_or_neg)

def classify(nr, train_data,train_target,test_data,test_target,val_data,val_target):
    print(nr)
    # vectorize data by fit and transform
    vectorizer = CountVectorizer(ngram_range = nr)
    train_data = vectorizer.fit_transform(train_data)
    val_data = vectorizer.transform(val_data)
    test_data = vectorizer.transform(test_data)

    # build the model with train_data and validate the model with val_data
    # get the alpha with max accuracy
    max_accu = 0
    max_alpha = 0
    for alpha in np.arange(0.05, 10, .05):
        nbc = MultinomialNB(alpha = alpha)
        nbc.fit(train_data, train_target) 
        predict = nbc.predict(val_data) 
        count = 0
        for left , right in zip(predict, val_target):
            if left == right:
                count += 1
        if(count/len(val_target) > max_accu):
            max_accu = count/len(val_target)
            max_alpha = alpha
    print(max_accu)

    # use the max accuracy to build the model and test it
    nbc = MultinomialNB(alpha = max_alpha)
    nbc.fit(train_data, train_target) 
    predict = nbc.predict(test_data) 
    count = 0
    for left , right in zip(predict, test_target):
        if left == right:
            count += 1
    print(count/len(test_target))


if __name__ == "__main__":
    train_data = []
    train_target = []
    val_data = []
    val_target = []
    test_data = []
    test_target = []

    # save data into list _data, save neg/pos into list _target
    for i in range(1,len(sys.argv)):
        f = open(sys.argv[i],'r') 
        file_name = sys.argv[i]
        pos_or_neg = 1 if 'pos' in file_name else 0

        if( "train" in file_name):
            save_into_list(f, train_data, pos_or_neg, train_target)
        elif( "test" in file_name):
            save_into_list(f, test_data, pos_or_neg, test_target)
        else:
            save_into_list(f, val_data, pos_or_neg, val_target)
        f.close()
    print("finish processing")
    

    # classify,validate and test for unigram, bigram, and unigram+bigram
    ngram_range = [(1,1),(2,2),(1,2)]
    for nr in ngram_range:
        classify(nr, train_data,train_target,test_data,test_target,val_data,val_target)

    

