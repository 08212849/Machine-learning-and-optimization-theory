import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional, GlobalAveragePooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import pickle
import warnings
warnings.filterwarnings("ignore")

def loadData():
    """
    加载数据集，train_texts_orig为一个含有2000条文本的list，其中前1000条文本为正面评价，后1000条为负面评价
    :return: 所有评价内容的list
    """
    pos_txts = os.listdir('./数据集/ChnSentiCorp_htl_ba_2000/pos')
    neg_txts = os.listdir('./数据集/ChnSentiCorp_htl_ba_2000/neg')
    train_texts_orig = [] # 存储所有评价，每例评价为一条string

    for i in range(len(pos_txts)):
        with open('./数据集/ChnSentiCorp_htl_ba_2000/pos/'+pos_txts[i], 'r', errors='ignore') as f:
            text = f.read().strip()
            train_texts_orig.append(text)
            f.close()
    for i in range(len(neg_txts)):
        with open('./数据集/ChnSentiCorp_htl_ba_2000/neg/'+neg_txts[i], 'r', errors='ignore') as f:
            text = f.read().strip()
            train_texts_orig.append(text)
            f.close()
    return train_texts_orig

def cutAndtokenize(train_texts_orig):
    """
    进行分词和tokenize
    :return: 每个样本的索引list集合
    """
    train_tokens = []
    for text in train_texts_orig:
        # 去掉标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）] + ", "", text)
        # 结巴分词
        cut = jieba.cut(text)
        # 结巴分词的输出结果为一个生成器,把生成器转换为list
        cut_list = [i for i in cut]
        for i, word in enumerate(cut_list):
            try:
                # 将词转换为索引index
                cut_list[i] = cn_model.key_to_index[word]
            except KeyError:
                # 如果词不在字典中，则输出0
                cut_list[i] = 0
        train_tokens.append(cut_list)
    return train_tokens

def drawTokensDistri():
    """
    确定tokens长度，绘制tokens长度分布图
    """
    # 平均tokens的长度
    print('平均tokens长度：', np.mean(num_tokens))
    # 最长的评价tokens的长度
    print('最大tokens长度：', np.max(num_tokens))
    max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))
    print('当tokens长度为 ', max_tokens, ' 时覆盖的样本比例：')
    print(np.sum( num_tokens < max_tokens ) / len(num_tokens))
    plt.hist(np.log(num_tokens), bins = 100)
    plt.xlim((0, 10))
    plt.ylabel('number of tokens')
    plt.xlabel('length of tokens')
    plt.title('Distribution of tokens length')
    plt.show()

def reverse_tokens(tokens):
    """
    用来将tokens转换为文本
    """
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index_to_key[i]
        # else:
        #     text = text + ' '
    return text

def embeddmatrixInit():
    """
    初始化embedding_matrix,一个 [num_words，embedding_dim] 的矩阵
    :return: embedding_matrix
    """
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for i in range(num_words):
        embedding_matrix[i, :] = cn_model[cn_model.index_to_key[i]]
    embedding_matrix = embedding_matrix.astype('float32')
    return embedding_matrix

def tokensPadding(train_tokens, max_tokens):
    """
    进行padding（填充）和truncating（修剪）,将索引长度标准化
    :return: train_padding数组
    """
    train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                            padding='pre', truncating='pre')
    # 超出五万个词向量的词用0代替
    train_pad[train_pad >= num_words] = 0
    return train_pad

def printPreposs():
    print('---------------------------------')
    print('查看训练样本，确认无误')
    print('原始文本：')
    print(train_texts_orig[30])
    print('数据预处理后一个样本的词向量表示：')
    print(train_pad[30])
    print('反向索引化，把索引转换成可阅读的文本：')
    print(reverse_tokens(train_pad[30]))
    print('label: ', train_target[30])
    print('---------------------------------')

def FNN():
    """
    num_words: 词汇表大小
    embedding_dim: 词向量维度
    embedding_matrix: (num_words, embedding_dim) 预训练词向量矩阵
    max_tokens: 输入序列的最大长度
    """
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=False))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))  # 64 神经元的隐藏层
    # model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))  # 32 神经元的隐藏层
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 输出层
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # 对于二分类问题，使用二元交叉熵损失函数
                  metrics=['accuracy'])
    return model

def BiLSTM():
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_tokens,
                        trainable=False))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(LSTM(units=16, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    # 我们使用adam以0.001的learning rate进行优化
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model

def callbackList():
    """
    EarlyStopping 用于在模型性能不再提升时提前终止训练
    ModelCheckpoint 用于在训练过程中保存模型的权重
    ReduceLROnPlateau 会在每次性能停止提升时减少学习率
    :return: 回调函数列表
    """
    # 建立一个权重的存储点
    path_checkpoint = 'sentiment_checkpoint.keras'
    checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                 verbose=1, save_weights_only=True,
                                 save_best_only=True)
    try:
        model.load_weights(path_checkpoint)
    except Exception as e:
        print(e)
    # early stoping 如果3个epoch内validation loss没有改善则停止训练
    earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1, min_lr=1e-5, patience=0,
                                           verbose=1)
    # 定义callback函数
    callbacks = [
        earlystopping,
        lr_reduction,
        # checkpoint
    ]
    return callbacks

def trainModel(model):
    print(model.summary())
    model.fit(X_train, y_train,
              validation_split=0.1,
              epochs=20,
              batch_size=128,
              callbacks=callbacks)
    result = model.evaluate(X_test, y_test)
    print('Accuracy:{0:.2%}'.format(result[1]))

embedding_dim = 300
num_words = 50000
max_tokens = 260
# 使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('./sgns.zhihu.bigram/sgns.zhihu.bigram',
                                          binary=False)
embedding_matrix = embeddmatrixInit()
if os.path.exists('train_pad.pkl'):
    # 文件存在，加载预处理后的数据
    with open('train_pad.pkl', 'rb') as file:
        train_pad = pickle.load(file)
else:
    train_texts_orig = loadData()
    train_tokens = cutAndtokenize(train_texts_orig)
    num_tokens = np.array([len(tokens) for tokens in train_tokens])
    max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))
    train_pad = tokensPadding(train_tokens, max_tokens)
    with open('train_pad.pkl', 'wb') as f:
        pickle.dump(train_pad, f)

# 准备target向量，前2000样本为1，后2000为0
train_target = np.concatenate((np.ones(1000), np.zeros(1000)))
X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=0.1, random_state=12)
model = BiLSTM()
callbacks = callbackList()
trainModel(model)

def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.key_to_index[word]
            if cut_list[i] >= 50000:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')
    # 预测
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)

test_list = [
    '酒店设施不是新的，服务态度很不好',
    '酒店卫生条件非常不好',
    '床铺非常舒适',
    '房间很凉，不给开暖气',
    '房间很凉爽，空调冷气很足',
    '酒店环境不好，住宿体验很不好',
    '房间隔音不到位' ,
    '晚上回来发现没有打扫卫生',
    '因为过节所以要我临时加钱，比团购的价格贵'
]
for text in test_list:
    predict_sentiment(text)

