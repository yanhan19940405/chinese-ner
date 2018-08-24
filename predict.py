import numpy as np
import main
import trainner
from keras.preprocessing.sequence import pad_sequences
np.set_printoptions(threshold=np.nan) #全部输出
batch_size=32
n_epoch=5
maxlen=trainner.maxlen
tag=trainner.tag
embedding_matrix=trainner.embedding_matrix
word_index=trainner.word_index
num_class=trainner.num_class
train_data=trainner.train_data
Y=trainner.Y
model,X_test,y_test=main.train_model(batch_size, n_epoch,embedding_matrix, word_index, num_class, train_data, tag,Y)
model.load_weights('./model/crf2.h5')
str = "2015年10月2日，来自印度的桑达尔·皮查伊（Sundar Pichai，昵称“劈柴”）刚刚登上谷歌的权力之巅，他作为CEO在接受福布斯采访时说：“我并不认为中国市场是一个黑洞。这是一个巨大的机遇，而我们可以扮演一种支持平台。未来，我们也有机会提供其他服务。”"
str1 = "他解释称，虽然谷歌先于百度早在2010年就面向中国市场推出独立搜索服务，但百度是“后来者居上”，“通过技术和产品创新反超Google”。并表示2010年谷歌退出中国市场时，百度在国内的市场份额已经超过70%，而谷歌的市场份额则在持续下降。"
strlist1 = main.loading_data(str,word_index)
strlist2 = main.loading_data(str1,word_index)
predict = [strlist1, strlist2]
test_data = pad_sequences(predict, maxlen=maxlen, padding="post")
a = model.predict(test_data)
print(a)
print(a.shape)
all_label=[]
predict_label=[]
tr=0
vb=0
c=0
for tr in range(len(a)):
    for vb in range(len(a[tr])):
        label_index=list(a[tr][vb]).index(1)
        predict_label.append(tag[label_index])
        if len(predict_label)==len(list(test_data[tr])):
            all_label.append(predict_label)
            predict_label = []
        vb=vb+1
    tr=tr+1
print(all_label)
print(1)
