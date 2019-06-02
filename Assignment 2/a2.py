import csv
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

def save_into_list(f, unigrams, bigrams, pos_or_neg, target):
    for line in f.readlines():
        line = line[1:-2].split(r", ")
        for i in range(len(line)):
            line[i] = line[i][1:-1]
            unigrams.append(line[i])
            target.append(pos_or_neg)
        bigrams.extend([i for i in zip(line[:-1],line[1:])])

if __name__ == "__main__":

    train_unigrams = []
    train_bigrams = []
    train_target = []
    val_unigrams = []
    val_bigrams = []
    val_target = []
    test_unigrams = []
    test_bigrams = []
    test_target = []

    for i in range(1,len(sys.argv)):
        f = open(sys.argv[i],'r') 
        feature_type = sys.argv[i][:2]
        pos_or_neg = 1 if sys.argv[i][-7:-4] == 'pos' else 0

        if( feature_type == 'tr'):
            save_into_list(f, train_unigrams, train_bigrams, pos_or_neg, train_target)
        elif( feature_type == 'te'):
            save_into_list(f, test_unigrams, test_bigrams, pos_or_neg, test_target)
        else:
            save_into_list(f, val_unigrams, val_bigrams, pos_or_neg, val_target)
        f.close()
    
    for alpha in range(1,20):
        nbc = Pipeline([
        ('vect', TfidfVectorizer( )),
        ('clf', MultinomialNB(alpha = alpha))])
        nbc.fit(train_unigrams, train_target) 
        predict = nbc.predict(test_unigrams) 
        count = 0
        for left , right in zip(predict, test_target):
            if left == right:
                count += 1
        print(str(alpha)+":")
        print(count/len(test_target))
