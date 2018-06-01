import numpy as np
import tensorflow as tf
import datetime # 使用Tensorboard来可视化损失值和正确率部分
import addition # 导入同文件夹下的addition.py（使用该python文件下的一些自定义函数）
# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息.
# 用户只需要封装任意的迭代器 tqdm(iterator)。
from tqdm import tqdm 
import time

"""参数设置"""
maxSeqLength = 30 # 设置句子长度上限（即包含词语个数上限）
numDimensions = 50 # 词向量的维度
batchSize = 20 # 批处理大小
lstmUnits = 64 # LSTM的单元个数
numClasses = 2 # 分类类别
# learning_rate=0.0001 # 优化函数的学习率
Train_rounds = 15 # 训练轮数。重复训练所以训练集的次数
Train_batch_iterations = 150 # 训练集，1轮的迭代次数
Test_batch_iterations  = 50 # 测试集，1轮的迭代次数
forget_bias_parameter = 1.0 # tf.contrib.rnn.BasicLSTMCell的forget_bias的值设置参数
# forget_bias_parameter：表示LSTM的cell单元遗忘门的偏置（默认设置1.0）
output_dropout = 0.9 # LSTM输出的dropout rate设置


"""导入索引字典"""
wordsList = np.load('./lib/wordsList.npy')
print('加载成功，数字索引词语! 这是一个包含', len(wordsList), '个单词的python列表')
wordsList = wordsList.tolist() # 原始数据以array形式加载，转为list形式
wordIndexVector = np.load('./lib/wordIndexVector.npy')
wordIndexVector = np.float32(wordIndexVector) # 将64为浮点数矩阵转换为32为浮点数矩阵float64->float32
print ('加载成功，数字索引词向量! 这是一个包含所有词向量值的', wordIndexVector.shape, '维的嵌入矩阵')

'''
我们也可以在词库中搜索词语，
比如 “好”，然后可以通过访问嵌入矩阵来得到相应的向量，如下：
wordIndex = wordsList.index('好')
wordVectors[wordIndex]
'''
'''
假设我们现在的输入句子是 “认为 好 没什么 问题”。
为了得到词向量，我们可以使用 TensorFlow 的嵌入函数。
这个函数有两个参数，一个是嵌入矩阵（在我们的情况下是词向量矩阵），另一个是每个词对应的索引。
具体的例子如下。

firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index('认为')
firstSentence[1] = wordsList.index('好')
firstSentence[2] = wordsList.index('没什么')
firstSentence[3] = wordsList.index('问题')
print(firstSentence.shape)
print(firstSentence)
输出数据是一个 10*50 的词矩阵，其中包括 10 个词，每个词的向量维度是 50。
with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordIndexVector,firstSentence).eval().shape)
'''

tf.reset_default_graph()


"""处理数据"""
X, Y = addition.import_data()
x_train, y_train, x_test, y_test = addition.one_three(X, Y)
# 对训练数据和测试数据分别进行补零
x_train,x_train_len = addition.zero_padding(x_train, maxSeqLength) 
x_test,x_test_len = addition.zero_padding(x_test, maxSeqLength)
print('训练集样本数：',len(x_train))
print('测试集样本数：',len(x_test))


"""定义占位符"""
with tf.name_scope('inputs'): 
    # 需要指定三个占位符
    # input_data用于数据输入。
    input_data = tf.placeholder(tf.int32, shape=[None, maxSeqLength], name='InputData')
    # 这个是评论文本的实际长度
    seq_len_ph = tf.placeholder(tf.int32,shape=None, name='Seq_len_ph')
    # labels用于标签数据。
    labels = tf.placeholder(tf.int32, shape=[None, numClasses], name='Labels')


"""初始化词向量"""
# 一旦，我们设置了我们的输入数据占位符，我们可以调用tf.nn.embedding_lookup() 函数来得到我们的词向量。
# 该函数最后将返回一个三维向量，第一个维度是批处理大小，第二个维度是句子长度，第三个维度是词向量长度。
with tf.name_scope('Embedding'):
    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordIndexVector,input_data)


"""定义lstm模型"""
# 使用 tf.nn.rnn_cell.BasicLSTMCell 函数，这个函数输入的参数是一个整数，表示需要几个 LSTM 单元。
# 设置一个dropout参数，以此来避免一些过拟合。
# 最后，我们将 LSTM cell 和三维的数据输入到 tf.nn.dynamic_rnn 
#     这个函数的功能是展开整个网络，并且构建一整个RNN模型。
with tf.name_scope('Lstm_model'):
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits, forget_bias=forget_bias_parameter)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=output_dropout)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, sequence_length=seq_len_ph, dtype=tf.float32)


"""输入全连接网络"""
# dynamic RNN 函数的第一个输出可以被认为是最后的隐藏状态向量。
# 这个向量将被重新确定维度，然后乘以最后的权重矩阵和一个偏置项来获得最终的输出值。
with tf.name_scope('Fully_connected'):
    with tf.name_scope('Weight'):
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        tf.summary.histogram('Fully_connected'+"/Weight",weight) # 直方图，可视化观看变量weight（Tensorboard）
    with tf.name_scope('Bias'):
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        tf.summary.histogram('Fully_connected'+"/Bias",bias) # 直方图，可视化观看变量bias（Tensorboard）
    with tf.name_scope('Wx_plus_b'):
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = tf.matmul(last, weight) + bias
        tf.summary.histogram('Fully_connected'+"/Wx_plus_b",prediction) # 直方图，可视化观看变量prediction（Tensorboard）
    with tf.name_scope('tanh_Wx_plus_b'):
        tanh_prediction = tf.nn.tanh(prediction)
        tf.summary.histogram('Fully_connected'+"/tanh_Wx_plus_b",tanh_prediction) # 直方图，可视化观看变量tanh_prediction（Tensorboard）


"""定义损失函数和优化器和精确率"""
# 接下来，我们需要定义正确的预测函数和正确率评估参数。
# 正确的预测形式是查看最后输出的0-1向量是否和标记的0-1向量相同。
with tf.name_scope('Accuracy'):
    correctPred = tf.equal(tf.argmax(tanh_prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
    tf.summary.scalar('Accuracy', accuracy) # 记录常量accuracy（Tensorboard）
# 之后，我们使用一个标准的交叉熵损失函数来作为损失值。
# 对于优化器，我们选择 Adam，并且采用默认的学习率。
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tanh_prediction, labels=labels))
    tf.summary.scalar('Loss', loss) # 记录常量loss（Tensorboard）

for learning_rate in [1E-4, 1E-5, 1E-6]:
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    
    """使用Tensorboard可视化tensorflow过程"""
    merged = tf.summary.merge_all() # 合并到Summary中
    
    
    """训练与测试"""
    # Train_batch_iterations = int(len(x_train)/batchSize) # 训练集批处理迭代次数
    # Test_batch_iterations = int(len(x_test)/batchSize) # 测试集批处理迭代次数
    Train_iterations = Train_batch_iterations*Train_rounds # 训练集总迭代次数
    with tf.Session() as sess:
        if learning_rate==1E-4:
            writer = tf.summary.FileWriter("./graphs"+learning_rate, sess.graph) # 选定可视化存储目录
            inti5 = tf.global_variables_initializer()
            sess.run(inti5)
        elif learning_rate==1E-5:
            writer_1 = tf.summary.FileWriter("./graphs"+learning_rate) # learning_rate==1E-5
        else:
            writer_2 = tf.summary.FileWriter("./graphs"+learning_rate) # learning_rate==1E-6
        Loss = 0
        Accuar = 0
        for k in range(Train_rounds):
            gen_train = addition.gen_batch(x_train, y_train, x_train_len, batchSize)
            for i in tqdm(range(Train_batch_iterations)):
                DATA_train = gen_train.__next__()
                list(DATA_train)
                train_data = DATA_train[0]
                train_label = DATA_train[1]
                train_seq = DATA_train[2]
                train_binary_label = addition.label(train_label)
                # 运行激活函数tanh
                sess.run(tanh_prediction,feed_dict={input_data: train_data, seq_len_ph: train_seq, labels: train_binary_label})
                # 运行损失函数
                loss_train, accuar_train = sess.run([loss, accuracy],feed_dict={input_data: train_data, seq_len_ph: train_seq, labels: train_binary_label})
                # 运行优化器
                sess.run(optimizer,feed_dict={input_data: train_data, seq_len_ph: train_seq, labels: train_binary_label})
                # merged也是需要run的（Tensorboard）
                result = sess.run(merged,feed_dict={input_data: train_data, seq_len_ph: train_seq, labels: train_binary_label})
                step = k*150+i
                if learning_rate==1E-4:
                    writer.add_summary(result, step) # result是summary类型的，需要放入writer中，step步数（x轴）
                elif learning_rate==1E-5:
                    writer_1.add_summary(result, step) # learning_rate==1E-5
                else:
                    writer_2.add_summary(result, step) # learning_rate==1E-6
                print('第', k, '轮：（共', Train_rounds, '轮）', loss_train,accuar_train)
                Loss += loss_train
                Accuar += accuar_train
                time.sleep(1)
            print("训练损失函数是: {:.3f}, 训练精确率是: {:.3f}".format(Loss/Train_iterations, Accuar/Train_iterations))
        
        gen_test = addition.gen_batch(x_test, y_test, x_test_len, batchSize)
        Loss = 0
        Accuar = 0
        for j in range(Test_batch_iterations):
            DATA_test = gen_test.__next__()
            list(DATA_test)
            test_data = DATA_test[0]
            test_label = DATA_test[1]
            test_seq = DATA_test[2]
            test_binary_label = addition.label(test_label)
            loss_test, accuar_test = sess.run([loss, accuracy],feed_dict={input_data: test_data, seq_len_ph: test_seq, labels: test_binary_label})
            print(loss_test, accuar_test)
            Loss += loss_test
            Accuar += accuar_test
            #writer = tf.summary.FileWriter("./graphs", sess.graph)
        print("测试损失函数是: {:.3f}, 测试精确率是: {:.3f}".format(Loss / Test_batch_iterations, Accuar / Test_batch_iterations))

writer.close()

