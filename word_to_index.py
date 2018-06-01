# -*- coding: utf-8 -*-
import numpy as np
wordsList = np.load('.\lib\wordsList.npy')
print('加载数字索引词语，成功!')
wordsList = wordsList.tolist() # Originally loaded as numpy array


'''生成积极文本词语索引'''
text_word_index=[]
with open('.\lib\Stop_WordsFilter_pos.txt', "r", encoding='utf-8') as f:
    for line in f.readlines():
        row_word_index=[]
        row_words = line.split()
        i = 0
        for word in row_words:
            if i == 0:
                row_word_index.append(wordsList.index(word.strip('\ufeff')))
                i = i + 1
            else:
                row_word_index.append(wordsList.index(word))
                i = i + 1
        text_word_index.append(row_word_index)
print ('积极文本词语索引矩阵构建，成功!')

'''存储积极文本词语索引'''
print ('积极索引矩阵导入processed_ID_pos.txt文本中。。。')
f=open('.\lib\processed_ID_pos.txt','w')
# 先存第0行的评论。
# .strip('[]')：用来删除文本中的中括号。
# .replace(',', '')：文本中的逗号用空格来替代
f.write(str(text_word_index[0]).strip('[]').replace(',', ''))
# 再存第1行至最后一行的评论，在存储每一行之前加换行符'\n'进行换行
for i in range(1, len(text_word_index)):
    f.write('\n'+str(text_word_index[i]).strip('[]').replace(',', ''))
f.close()


'''生成消极文本词语索引'''
text_word_index=[]
with open('.\lib\Stop_WordsFilter_neg.txt', "r", encoding='utf-8') as f:
    for line in f.readlines():
        row_word_index=[]
        row_words = line.split()
        i = 0
        for word in row_words:
            if i == 0:
                row_word_index.append(wordsList.index(word.strip('\ufeff')))
                i = i + 1
            else:
                row_word_index.append(wordsList.index(word))
                i = i + 1
        text_word_index.append(row_word_index)
print ('消极文本词语索引矩阵构建，成功!')

'''存储消极文本词语索引'''
print ('消极索引矩阵导入processed_ID_neg.txt文本中。。。')
f=open('.\lib\processed_ID_neg.txt','w')
f.write(str(text_word_index[0]).strip('[]').replace(',', ''))
for i in range(1, len(text_word_index)):
    f.write('\n'+str(text_word_index[i]).strip('[]').replace(',', ''))
f.close()
print ('存储完成!')