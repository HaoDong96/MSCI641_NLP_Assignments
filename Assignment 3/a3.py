from gensim.models import word2vec

def save_into_list(f, data):
    # save each line into list
    for line in f.readlines():
        data.append(eval(line))

if __name__ == "__main__":
    #read data from files
    data = []
    for f_name in ["training_pos.csv", "training_neg.csv", "validation_pos.csv", 
                    "validation_neg.csv", "test_pos.csv", "test_neg.csv"]:
        f = open(f_name,'r') 
        save_into_list(f, data)

    # train the model
    model =  word2vec.Word2Vec(data, size=300)

    print("20 most similar words to 'good':")
    y1 = model.wv.most_similar("good", topn = 20)
    for item in y1:
        print(item[0],item[1])
    
    print("20 most similar words to 'bad':")
    y2 = model.wv.most_similar("bad", topn = 20)
    for item in y2:
        print(item[0],item[1])