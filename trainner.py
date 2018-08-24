import numpy as np
import main

np.set_printoptions(threshold=np.nan) #全部输出
path = "./data/train.txt"
data=main.writedata(path)
k=30000
sen,sen_label=main.dealing_data(data)
main.train_data_vector(sen)
maxlen, sen_label, k=main.label_dev(sen, sen_label, k)
Y, tag, embedding_matrix, word_index, num_class, train_data=main.y_label_change(k, maxlen, sen_label, sen)
batch_size=32
n_epoch=5
model,X_test,y_test=main.train_model(batch_size, n_epoch, embedding_matrix, word_index, num_class, train_data, tag,Y)
predicted=main.test(model,X_test,y_test,tag,batch_size)
print(predicted)
print(1)
