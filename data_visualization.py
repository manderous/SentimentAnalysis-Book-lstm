numWords = []
with open('.\lib\Stop_WordsFilter_pos.txt', "r", encoding='utf-8') as f:
    for line in f.readlines():
        counter = len(line.split())
        numWords.append(counter)
print('统计正向文本中，每句话中词语的个数，完成！保存在numWords变量中')

with open('.\lib\Stop_WordsFilter_neg.txt', "r", encoding='utf-8') as f:
    for line in f.readlines():
        counter = len(line.split())
        numWords.append(counter)
print('统计负向文本中，每句话中词语的个数，完成！保存在numWords变量中')

print('一共有', len(numWords), '条书籍评论语料')
print('在书籍评论语料中，一共有', sum(numWords), '个词语')
print('在书籍评论语料中，每句话平均有', sum(numWords)/len(numWords), '个词语')

print('在书籍评论语料中，可视化一条评论所含词语个数的分布如下图所示：')
'''使用Matplot将数据进行可视化'''
import matplotlib.pyplot as plt
plt.hist(numWords, 50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axis([0, 300, 0, 1400])
plt.show()

'''
据直方图和句子的平均单词数，句子最大长度暂时设置为25
'''
