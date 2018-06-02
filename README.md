# SentimentAnalysis-Book-lstm

这里是利用python3.6搭建tensorflow框架编程实现的一层、两层以及双向LSTM模型，且最终可在tensorbosrd上查看实验结果的文件。README.txt文件按照实验先后顺序，介绍了各文件。如需进行实验，可按照以下步骤进行。
其中：<br>
* （1）-（4）：数据预处理<br>
* （5）-（8）：一层、两层以及双向lstm模型<br>
* （9）-（12）：灵敏度分析<br>
* （13）：lib文件夹下的数据文件介绍<br>

****

|Author|manderous|
|---|---|
|E-mail|manderous@foxmail.com|

****

## 目录
* [数据预处理](#数据预处理)
    * [(1)data_visualization.py](#(1)data_visualization.py)
    * [(2)word_to_index.py](#(2)word_to_index.py)
    * [(3)word2vec_test.py](#(3)word2vec_test.py)
    * [(4)zhwiki_2017_03.sg_50d.word2vec](#(4)zhwiki_2017_03.sg_50d.word2vec)
* [LSTM模型](#模型)
    * [(5)lstm_test.py](#(5)lstm_test.py)
    * [(6)lstm_multi_test.py](#(6)lstm_multi_test.py)
    * [(7)lstm_bi_test.py](#(7)lstm_bi_test.py)
    * [(8)addition.py](#(8)addition.py)
* [灵敏度分析](#灵敏度分析)
    * [(9)pp_bi.py](#(9)pp_bi.py)
    * [(10)pp.py](#(10)pp.py)
    * [(11)pp_multi_lstmunits.py](#(11)pp_multi_lstmunits.py)
    * [(12)pp_multi_maxseqlen.py](#(12)pp_multi_maxseqlen.py)
* [(13)lib文件夹下的数据文件介绍](#(13)lib文件夹下的数据文件介绍)

****

## 数据预处理

### (1)data_visualization.py
#### 可视化句子长度的分布（python文件）
输入：<br>
.\lib\Stop_WordsFilter_pos.txt：最初的笔记本评论-积极文本-经过分词、停用词过滤得到的文本文件<br>
.\lib\Stop_WordsFilter_neg.txt：最初的笔记本评论-消极文本-经过分词、停用词过滤得到的文本文件<br>

### (2)word_to_index.py
#### 生成积极文本和消极文本的词语索引，保存至文本文件中（python文件）
输入：<br>.
\lib\Stop_WordsFilter_pos.txt：最初的笔记本评论-积极文本-经过分词、停用词过滤得到的文本文件<br>
.\lib\Stop_WordsFilter_neg.txt：最初的笔记本评论-消极文本-经过分词、停用词过滤得到的文本文件<br>
输出：<br>
.\lib\processed_ID_neg.txt：积极文本的词语索引<br>
.\lib\processed_ID_pos.txt：消极文本的词语索引<br>

### (3)word2vec_test.py
#### 导入搜狗词向量语料加载，将积极文本和消极文本的词语生成词向量，保存至文本文件中（python文件）
输入：<br>
.\lib\Stop_WordsFilter_pos.txt：最初的笔记本评论-积极文本-经过分词、停用词过滤得到的文本文件<br>
.\lib\Stop_WordsFilter_neg.txt：最初的笔记本评论-消极文本-经过分词、停用词过滤得到的文本文件<br>
输出：<br>
.\lib\wordsList.npy：数字索引词语变量<br>
.\lib\wordIndexVector.npy：数字索引词向量变量<br>

### (4)zhwiki_2017_03.sg_50d.word2vec
#### 搜狗词向量语料（word2vec文件）
百度云资源：https://pan.baidu.com/s/1C94HXCCWOmX-W4IbajXFyA

****

## 模型

### (5)lstm_test.py
#### lstm模型（python文件）
可在tensorboard上查看实验结果<br>
tensorboard启动方法：<br>
<1>首先cmd找到命令提示符的对话框；<br>
<2>然后cd切换自己的tensorboard所在的文件路径（我的tensorboard路径是：Anaconda3\envs\tensorflow-gpu\Scripts）；<br>
<3>（如果当前的python不是在tensorflow环境下）执行命令activate tensorflow-gpu（我的是gpu版本的tensorflow）；<br>
<4>执行命令tensorboard --logdir=logs（其中logs=tensorflow代码运行生成的events文件，当logs 中有多个events时，会生成tensorboard的scalar 的对比图，但tensorboard的graph只会展示最新的结果）；<br>
<5>把命令提示符的对话框最终生成的网址(http://DESKTOP-S2Q1MOS:6006， 每个人的可能不一样)复制到浏览器中打开即可。<br>

### (6)lstm_multi_test.py
#### 两层lstm模型（python文件）
可在tensorboard上查看实验结果

### (7)lstm_bi_test.py
#### 双向lstm模型（python文件）
可在tensorboard上查看实验结果

### (8)addition.py
#### 改python文件下自定义了一些函数，方便lstm模型模型使用（python文件）

****

## 灵敏度分析

### (9)pp_bi.py
#### 关于“双向lstm模型”的灵敏度分析（对学习率的灵敏度分析：learning_rate=1E-4；1E-5；1E-（python文件）

### (10)pp.py
#### 关于“lstm模型”的灵敏度分析（对学习率的灵敏度分析：learning_rate=1E-4；1E-5；1E-6）（python文件）

### (11)pp_multi_lstmunits.py
#### 关于“两层lstm模型”的灵敏度分析（对lstm单元个数的灵敏度分析：lstmUnits=50；70；100；110；120；130；140；150；180；200）（python文件）

### (12)pp_multi_maxseqlen.py
#### 关于“两层lstm模型”的灵敏度分析（对句子长度上限的灵敏度分析：maxSeqLength=25；36；38；40；43；45；47；49；55；60）（python文件）

****

### (13)lib文件夹下的数据文件介绍：
Stop_WordsFilter_pos.txt：最初的笔记本评论-积极文本-经过分词、停用词过滤得到的文本文件（txt文件）<br>
Stop_WordsFilter_neg.txt：最初的笔记本评论-消极文本-经过分词、停用词过滤得到的文本文件（txt文件）<br>
processed_ID_neg.txt：积极文本的词语索引（word_to_index.py生成的 txt文件）<br>
processed_ID_pos.txt：消极文本的词语索引（word_to_index.py生成的 txt文件）<br>
wordsList.npy：数字索引词语变量（word2vec_test.py生成的 npy文件）<br>
wordIndexVector.npy：数字索引词向量变量（word2vec_test.py生成的 npy文件）<br>

### (end)
