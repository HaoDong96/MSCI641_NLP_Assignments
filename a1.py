import re
from nltk.corpus import stopwords
import random




def main():
    neg_stopwith = []
    neg_stopless = []

    neg_stopwith_train = []
    neg_stopwith_vali = []
    neg_stopwith_test = []
    neg_stopless_train = []
    neg_stopless_vali = []
    neg_stopless_test = []


    with open("../neg.txt") as f:
        stop_words = set(stopwords.words('english'))
        for line in f:
            for t in line.split(" "):
                t = re.sub(r"[!\"#$%&\(\)\*\+/:;<=>@\[\\\]^`{|}~(\t)(\n)]","",t)
                if t != "":
                    if t not in stop_words:
                        neg_stopless.append(t)
                    neg_stopwith.append(t)

        for token in neg_stopwith:
            random_factor = random.randint(0,9)
            if(random_factor < 8):
                neg_stopwith_train.append(token)
            elif(random_factor < 9):
                neg_stopwith_vali.append(token)
            else:
                neg_stopwith_test.append(token)

        print(len(neg_stopwith))
        print(len(neg_stopwith_train))
        print(len(neg_stopwith_vali))
        print(len(neg_stopwith_test))

        for token in neg_stopless:
            random_factor = random.randint(0,9)
            if(random_factor < 8):
                neg_stopless_train.append(token)
            elif(random_factor < 9):
                neg_stopless_vali.append(token)
            else:
                neg_stopless_test.append(token)

        print(len(neg_stopless))
        print(len(neg_stopless_train))
        print(len(neg_stopless_vali))
        print(len(neg_stopless_test))


if __name__ == "__main__":
    main()