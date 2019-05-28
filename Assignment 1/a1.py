import sys
import numpy as np
from nltk import download
from nltk.corpus import stopwords
import re


if __name__ == "__main__":
    download('stopwords')
    input_path = sys.argv[1]

    """
    Tokenize the input file here
    Create train, val, and test sets
    """
    no_stopword_list = []
    with_stopword_list = []
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
            no_stopword_list.append(no_stopword)
            with_stopword_list.append(with_stopword)
        

        np.random.shuffle(no_stopword_list)
        np.random.shuffle(with_stopword_list)

        length = len(with_stopword_list)
        train_list = with_stopword_list[:int(length*0.8)]
        val_list = with_stopword_list[int(length*0.8):int(length*0.9)]
        test_list = with_stopword_list[int(length*0.9):]

        length = len(no_stopword_list)
        train_list_no_stopword = no_stopword_list[:int(length*0.8)]
        val_list_no_stopword = no_stopword_list[int(length*0.8):int(length*0.9)]
        test_list_no_stopword = no_stopword_list[int(length*0.9):]

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
