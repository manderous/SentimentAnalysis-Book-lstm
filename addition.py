import numpy as np

# 设置句子的最大长度lenth，将超过该句子长度的句子截断，不足的句子补零。
def zero_padding(x_data,lenth):
    data = []
    length = []
    for i in x_data:
        if len(i) >= lenth:
            tep1 = i[0:lenth]
            tep2 = lenth
        else:
            tep1 = i+(lenth-len(i))*[0]
            tep2 = len(i)
        data.append(tep1)
        length.append(tep2)
    return data,length

# 打标签，y=0时（1，0），y=1时（0，1）
def label(y):
    data = np.zeros((len(y),2))
    for i in range(len(y)):
        if y[i]==0:
            data[i][0] = 1
            data[i][1] = 0
        else:
            data[i][0] = 0
            data[i][1] = 1
    return data


def gen_batch(Data,Label,Lenth,batch_size):

    Data = np.array(Data)
    Label = np.array(Label)
    Lenth = np.array(Lenth)

    size = len(Data)
    index1 = np.arange(size)
    np.random.shuffle(index1)
    Data = Data[index1]
    Label = Label[index1]
    Lenth = Lenth[index1]
    i = 0

    while 1>0:
        if i+batch_size <= size:
            yield Data[i:i+batch_size],Label[i:i+batch_size],Lenth[i:i+batch_size]
            i += batch_size
        else:
            yield Data[i:size],Label[i:size],Lenth[i:size]
            continue

# 导入数据
def import_data():
    # with open('pos_processed_ID.txt', 'r') as pos_data:
    with open('.\lib\processed_ID_pos.txt', 'r') as pos_data:
        pos_list = pos_data.readlines()
    # with open('neg_processed_ID.txt', 'r') as neg_data:
    with open('.\lib\processed_ID_neg.txt', 'r') as neg_data:
        neg_list = neg_data.readlines()
    pos = []
    for i in pos_list:
        tep = list(map(int, i.split(' ')[:-1]))
        pos.append(tep)

    neg = []
    for i in neg_list:
        tep = list(map(int, i.split(' ')[:-1]))
        neg.append(tep)

    label_pos = np.ones(len(pos), dtype=np.int)
    label_neg = np.zeros(len(neg), dtype=np.int)
    label_pos = label_pos.tolist()
    label_neg = label_neg.tolist()

    pos.extend(neg)
    label_pos.extend(label_neg)

    return pos,label_pos

# 分成8份，7份作训练集，1份作测试集
def one_three(x,y):
    x = np.array(x)
    y = np.array(y)
    size = len(x)
    index1 = np.arange(size)
    np.random.shuffle(index1) # 打乱顺序函数
    Data = x[index1]
    Label = y[index1]

    size_one = int(size/8)
    size_three = size-size_one
    train_Data = Data[0:size_three]
    train_label = Label[0:size_three]
    test_Data = Data[size_three:size]
    test_label = Label[size_three:size]

    return train_Data.tolist(), train_label.tolist(), test_Data.tolist(), test_label.tolist()

