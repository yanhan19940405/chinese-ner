# chinese-ner
##注意##
由于github某些组件最近被墙，导致图片无法正常加载，请按照如下方式处理：

1、更改hosts文件，添加如下信息，刷新网页即可解决此问题：

>> ```
>> 52.74.223.119 github.com
>> 192.30.253.119 gist.github.com
>> 54.169.195.247 api.github.com
>> 185.199.111.153 assets-cdn.github.com
>> 151.101.76.133 raw.githubusercontent.com
>> 151.101.108.133 user-images.githubusercontent.com
>> 151.101.76.133 gist.githubusercontent.com
>> 151.101.76.133 cloud.githubusercontent.com
>> 151.101.76.133 camo.githubusercontent.com
>> 52.74.223.119 github.com
>> 192.30.253.119 gist.github.com
>> 54.169.195.247 api.github.com
>> 185.199.111.153 assets-cdn.github.com
>> 151.101.76.133 raw.githubusercontent.com
>> 151.101.108.133 user-images.githubusercontent.com
>> 151.101.76.133 gist.githubusercontent.com
>> 151.101.76.133 cloud.githubusercontent.com
>> 151.101.76.133 camo.githubusercontent.com
>> ```

2、可以将源码图片文件夹中找到相应图片。

中文命名实体识别
# 注意
# 1.本案例使用字嵌入思想開發,使用word2vecu訓練樣本中所有字得到字典,從而進行字向量嵌入使用.
# 2.本案裏中採用BIO標注體系進行標注.
# 3.樣本數據分類達到16類,分類報告如下:
![图1 classreport](https://github.com/yanhan19940405/chinese-ner/blob/master/images/webwxgetmsgimg.jpeg)
# 4.在code中輸入句子進行預測結果如下圖:
![图2 predict](https://github.com/yanhan19940405/chinese-ner/blob/master/images/1586498115.jpg)
![图3 predict](https://github.com/yanhan19940405/chinese-ner/blob/master/images/1781764351.jpg)
實體基本識別完畢,但是模型還可以進一步改進,減少字的bio標注類別等等.
# 5.數據文件刪除,替換成自己的數據集可,數據集結構應該都懂得####
