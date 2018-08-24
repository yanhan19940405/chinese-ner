import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from gensim.models import word2vec
import gensim
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.corpora import LowCorpus
from gensim.corpora import Dictionary
import re
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras import optimizers
from keras import backend as K
np.set_printoptions(threshold=np.nan) #全部输出

def writedata(path):
    f = open(path, encoding='utf-8')
    data = []
    label = []
    datalist = []
    i = 0
    k = 0
    line = f.readline()
    while line and k >= 0:
        # print(line)
        data.append(line)
        line = f.readline()
        k = k + 1
    f.close()
    return data
def dealing_data(data):
    label = []
    sen=[]
    sen_len=[]
    sen_label=[]
    sum_sen=""
    for i in range(len(data)):
        if data[i][0] in list('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'):
            pass
        else:
            try:
                if data[i]!="\n":
                    sum_sen=sum_sen+str(data[i][0].replace('\n',''))
                    label.append(data[i][1:].replace('\n','').replace(' ',''))
                elif data[i]=="\n":
                    sen.append(list(sum_sen))
                    sen_label.append(label)
                    sum_sen=""
                    label=[]
                    # print("句子加载完毕")
            except IndexError:
                if data[i]!="\n":
                    sum_sen=sum_sen+''
                    label.append("0")
                elif data[i]=="\n":
                    sen.append(list(sum_sen))
                    sen_label.append(label)
                    sum_sen =""
                    label = []
                    # print("句子加载完毕")
        i=i+1#这里是按句子处理情况
    # print(sen)
    print("句子加载完毕")
    return sen,sen_label
def train_data_vector(sen):
    w1 = 0
    w2 = 0
    ARRS = []
    f = open('./data/ner.txt', 'w+', encoding="utf-8")
    for w1 in range(len(sen)):
        jointsFrame = sen[w1]  # 每行
        ARRS.append(jointsFrame)
        for w2 in range(len(sen[w1])):
            strNum = str(jointsFrame[w2])
            f.write(strNum)
            f.write(' ')
            w2 = w2 + 1
        f.write('\n')
        w1 = w1 + 1
    f.close()
    sentence = word2vec.Text8Corpus("./data/ner.txt")
    model = word2vec.Word2Vec(sentence, window=5, min_count=1, workers=4)
    model.save("./model/ner.model")
    print("字向量模型训练完毕")
def label_dev(sen,sen_label,k):
    m1 = 0
    m2 = 0
    sen_len=[]
    # for m1 in range(len(sen_label)):
    #     for m2 in range(len(sen_label[m1])):
    #         if sen_label[m1][m2] == "B-TIM":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "I-TIM":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "B-LOC":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "I-LOC":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "B-PRO":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "I-PRO":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "B-JOB":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "I-JOB":
    #             sen_label[m1][m2] = "O"
    #         elif sen_label[m1][m2] == "B-COM":
    #             sen_label[m1][m2] = "B-ORG"
    #         elif sen_label[m1][m2] == "I-COM":
    #             sen_label[m1][m2] = "I-ORG"
    #         m2 = m2 + 1
    #     m1 = m1 + 1
    # print(sen_label)  # 打印数据集中每一个句子中单字构成的BIO标签集合，呈二维结构
    for item in sen:
        sen_len.append(len(item))
        if len(sen_len) > k:
            break
    maxlen = max(list(sen_len))
    return maxlen,sen_label,k
def y_label_change(k,maxlen,sen_label,sen):
    sen=sen[0:k]
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(sen)   #texts作为处理对象
    word_sequence = tokenizer.texts_to_sequences(sen)  #将文本转换为由索引表示的序列数据
    train_data = pad_sequences(word_sequence, maxlen=maxlen, padding="post")    #对padding进行设置，否则会默认从前面开始填充
    word_index = tokenizer.word_index   #word到索引的映射列表
    # word_index['PAD']=0
    # word_index['UNK']=1
    print(train_data.shape)
    print("word",word_index)
    #model =  KeyedVectors.load_word2vec_format('./model/text.model.bin', binary=True)
    model=gensim.models.Word2Vec.load('./model/ner.model')
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    for word, i in word_index.items():
        if word in model:
            embedding_matrix[i] = np.asarray(model[word])
        elif word not in model:
                # words not found in embedding index will be all-zeros.
            embedding_matrix[i] =np.asarray(0)

    print("t",embedding_matrix.shape)
    tag=['O','B-TIM', 'I-TIM','B-LOC', 'I-LOC','B-ORG', 'I-ORG','B-COM', 'I-COM','B-PRO', 'I-PRO','B-JOB', 'I-JOB','B-PER', 'I-PER']
    Y =sen_label[0:k]#以下操作将二维结构的标签矩阵转换为多类别数值特征
    l=[]
    a=0
    b=0
    for a in range(len(Y)):
        for b in range(len(Y[a])):
            if Y[a][b]==tag[0]:
                Y[a][b]=0
            elif Y[a][b]==tag[1]:
                Y[a][b] = 1
            elif Y[a][b]==tag[2]:
                Y[a][b] = 2
            elif Y[a][b]==tag[3]:
                Y[a][b] = 3
            elif Y[a][b]==tag[4]:
                Y[a][b] = 4
            elif Y[a][b] == tag[5]:
                Y[a][b] = 5
            elif Y[a][b] == tag[6]:
                Y[a][b] = 6
            elif Y[a][b] == tag[7]:
                Y[a][b] = 7
            elif Y[a][b] == tag[8]:
                Y[a][b] = 8
            elif Y[a][b] == tag[9]:
                Y[a][b] = 9
            elif Y[a][b] == tag[10]:
                Y[a][b] = 10
            elif Y[a][b] == tag[11]:
                Y[a][b] = 11
            elif Y[a][b] == tag[12]:
                Y[a][b] = 12
            elif Y[a][b] == tag[13]:
                Y[a][b] = 13
            elif Y[a][b] == tag[14]:
                Y[a][b] = 14
            else:
                pass
            b=b+1
        a=a+1
    print(Y)
    Y=pad_sequences(Y, maxlen=maxlen, padding="post")
    # print("labelsall",np.array(labels_all).shape)
    num_class=len(set(list(tag)))
    print(num_class)
    Y=np.expand_dims(Y, 2)
    return Y,tag,embedding_matrix,word_index,num_class,train_data
def train_model(batch_size, n_epoch, embedding_matrix, word_index, num_class, train_data, tag,Y):
    embedding_layer = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix])
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.4, activation='tanh')))
    # model.add(Dropout(0.4))
    # model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.4, activation='tanh')))
    # model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(num_class)))
    # model.add(Dropout(0.4))
    # model.add(Dense(num_class, activation='softmax'))
    crf_layer = CRF(num_class, sparse_target=True)
    model.add(crf_layer)
    model.summary()
    optimizer = optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    X_train, X_test, y_train, y_test = train_test_split(train_data, Y, test_size=0.6)
    model.fit(X_train, y_train, verbose=1, batch_size=batch_size, epochs=n_epoch, validation_data=[X_test, y_test])
    model.save_weights('./model/crf2.h5')
    return model,X_test,y_test
def test(model,X_test,y_test,tag,batch_size):
    score, acc = model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
    test = [score, acc]
    print('Test score:', score)
    print('Test accuracy:', acc)
    predicted = model.predict(X_test)
    print("=============================================================================================")
    print("pre", predicted)
    print("=============================================================================================")
    tag1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    pre_label = []
    pre_label_all = []
    for bt5 in predicted:
        for bt7 in bt5:
            pre_label.append(tag1[np.argmax(bt7)])
        pre_label_all.append(pre_label)
        pre_label = []
    y_true = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
    y_true = [i for item in y_true for i in item]
    y_pred = [i for item in pre_label_all for i in item]
    target_names = tag
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(predicted.shape)
    return predicted
def loading_data(str,word_index):
    strlist = []
    for con in str:
        if con in word_index.keys():
            strlist.append(word_index[con])
        elif con not in word_index.keys():
            strlist.append(0)
    return strlist
# if __name__ == '__main__':
#     path = "./data/train.txt"
#     data=writedata(path)
#     k=100
#     sen,sen_label=dealing_data(data)
#     maxlen, sen_label, k=label_dev(sen, sen_label, k)
#     Y, tag, embedding_matrix, word_index, num_class, train_data=y_label_change(k, maxlen, sen_label, sen)
#     batch_size=32
#     n_epoch=5
#     model,X_test,y_test=train_model(batch_size, n_epoch, embedding_matrix, word_index, num_class, train_data, tag)
#     predicted=test(model,X_test, y_test)
#     print(predicted)
#     model.load_weights("./model/crf2.h5")
#     str = "根据2017年财报显示，腾讯公司亏损一半业绩，公司董事马化腾宣布重要对策。"
#     str1="中国社科院金融所银行研究室主任曾刚认为，收割韭菜的时候来到了。"
#     # str2="在财税专家张连起看来，面对复杂严峻的国内外经济形势，民众压力很大。"
#     # predict=[list(str),list(str1),list(str2)]
#     # tokenizer=Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
#     # tokenizer.fit_on_texts(predict)   #texts作为处理对象
#     # word_sequence = tokenizer.texts_to_sequences(predict)  #将文本转换为由索引表示的序列数据
#     # test_data = pad_sequences(word_sequence, maxlen=maxlen, padding="post")
#     # a=model.predict(test_data)
#     strlist1 = loading_data(str)
#     strlist2 = loading_data(str1)
#     predict = [strlist1, strlist2]
#     test_data = pad_sequences(predict, maxlen=maxlen, padding="post")
#     a = model.predict(test_data)
#     print(a)
#     print(a.shape)
#     all_label=[]
#     predict_label=[]
#     tr=0
#     vb=0
#     c=0
#     for tr in range(len(a)):
#         for vb in range(len(a[tr])):
#             label_index=list(a[tr][vb]).index(1)
#             predict_label.append(tag[label_index])
#             if len(predict_label)==len(list(test_data[tr])):
#                 all_label.append(predict_label)
#                 predict_label = []
#             vb=vb+1
#         tr=tr+1
#     print(1)
