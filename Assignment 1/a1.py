import sys
import numpy as np
from nltk import download
from nltk.corpus import stopwords
import re
import random


if __name__ == "__main__":
    download('stopwords')
    input_path = sys.argv[1]

    """
    Tokenize the input file here
    Create train, val, and test sets
    """
    train_list = []
    val_list = []
    test_list = []
    train_list_no_stopword = []
    val_list_no_stopword = []
    test_list_no_stopword = []

    with open(input_path) as f:
        stop_words = set(stopwords.words('english'))
        for line in f:
            no_stopword = []
            with_stopword = []
            line = re.sub(r"[!\"#$%&\(\)\*\+/:;<=>@\[\\\]^`{|}~(\t)(\n)]"," ",line).lower()
            # for t in line.split(" "):
            for t in list(filter(lambda x: x!=" " and x!="" and x!=None,re.split("(-+)(?!-)|([\W])",line))):
                if t != "":
                    if t not in stop_words:
                        no_stopword.append(t)
                    with_stopword.append(t)
            random_factor = random.randint(0,9)
            if(random_factor < 8):
                train_list.append(with_stopword)
                train_list_no_stopword.append(no_stopword)
            elif(random_factor < 9):
                val_list.append(with_stopword)
                val_list_no_stopword.append(no_stopword)
            else:
                test_list.append(with_stopword)
                test_list_no_stopword.append(no_stopword)

    print("finish proccessing")
    f.close()
    # sample_tokenized_list = [["Hetest_list_no_stopwordllo", "World", "."], ["Good", "bye"]]

    suffix = input_path[-7:-4]

    np.savetxt("train_"+ suffix+ ".csv", train_list, delimiter=",", fmt='%s')
    np.savetxt("val_"+ suffix+ ".csv", val_list, delimiter=",", fmt='%s')
    np.savetxt("test_"+ suffix+ ".csv", test_list, delimiter=",", fmt='%s')

    np.savetxt("train_no_stopword_"+ suffix+ ".csv", train_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("val_no_stopword_"+ suffix+ ".csv", val_list_no_stopword,
               delimiter=",", fmt='%s')
    np.savetxt("test_no_stopword_"+ suffix+ ".csv", test_list_no_stopword,
               delimiter=",", fmt='%s')
