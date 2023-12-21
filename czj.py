import json
import re
import time
import gensim
import gensim.models.word2vec as w2v
from gensim.corpora.dictionary import Dictionary
import jieba
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
import torch
import numpy as np


def clean(data):
    data_out = []

    # 清洗数据
    for temp in data:
        temp_json = json.loads(temp)
        # 数据清洗
        temp_json['content'] = re.sub(r'\/\/\@.*?(\：|\:)', "", temp_json['content'])  # 清除@用户信息
        temp_json['content'] = re.sub(r'\#.*?\#', "", temp_json['content'])  # 清除话题信息
        temp_json['content'] = re.sub(r'\【.*?\】', "", temp_json['content'])  # 清除话题信息
        temp_json['content'] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', "", temp_json['content'],flags=re.MULTILINE) # 清除链接信息

        # 转化label
        if (temp_json['label'] == 'neural'):
            temp_json['label'] = 0
        elif (temp_json['label'] == 'happy'):
            temp_json['label'] = 1
        elif (temp_json['label'] == 'angry'):
            temp_json['label'] = 2
        elif (temp_json['label'] == 'sad'):
            temp_json['label'] = 3
        elif (temp_json['label'] == 'fear'):
            temp_json['label'] = 4
        elif (temp_json['label'] == 'surprise'):
            temp_json['label'] = 5

        data_out.append(temp_json)

    return data_out


def load_deal_data(path):
    # 根据路径加载数据
    file = open('virus_train.txt', 'r', encoding='utf-8')
    string_raw = file.read()
    file.close()

    pattern = re.compile(r'{.+?}')  # 正则表达式，匹配字典
    data_load = pattern.findall(string_raw)
    data_raw = clean(data_load)
    return data_raw


data_raw_train = load_deal_data('virus_eval_labeled.txt')
data_raw_test = load_deal_data('virus_train.txt')


def splite_content(data):
    string_splite = ''
    pure_data = []
    for temp in range(len(data)):
        content = data[temp]['content']
        new_content = jieba.cut(content,cut_all=False)
        str_out = ' '.join(new_content)
        cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9^\s]") # 去除非中英文字、数字的所有字符
        str_out = cop.sub('', str_out)

        for i in range(6):
            str_out = str_out.replace('  ',' ') # 去除多余空格
        str_out = str_out.strip() # 去除两边空格

        data[temp]['content'] = str_out.split(' ')
        pure_data.append([data[temp]['content'],data[temp]['label']])
        str_out += '\r\n'
        string_splite += str_out
    return pure_data, string_splite

splite_word_all = '' # 分词总文本(包括训练文本和测试文本)
data_seg_train, out = splite_content(data_raw_train)
splite_word_all += out
data_seg_test, out = splite_content(data_raw_test)
splite_word_all += out

# 保存分好词的文本
f = open('splite_word_all.txt', 'w', encoding='utf-8')
f.write(splite_word_all)
f.close()


def analyse_word_num(data):
    data_num_train = len(data)  # 数据条数
    word_num = 0  # 总词数
    single_num = []  # 每条数据的长度的大小数组
    ave_num = 0  # 平均每条数据的词数大小

    for i in range(len(data)):
        single_num.append(len(data[i][0]))
        word_num += len(data[i][0])
    ave_num = word_num / data_num_train
    print('全部数据词总数为：', word_num, '; 每条数据的平均词数为：', ave_num)


    plt.hist(single_num, bins=500)
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.axis([0, 100, 0, 2500])
    plt.show()


analyse_word_num(data_seg_train)


#模型训练，生成词向量
model_file_name = 'model.txt'
sentences = w2v.LineSentence('splite_word_all.txt')
model = w2v.Word2Vec(sentences, size=30, window=20, min_count=5, workers=4) # 参数含义：数据源，生成词向量长度，时间窗大小，最小词频数，线程数
model.save(model_file_name)

# 使用训练好的模型
model = w2v.Word2Vec.load(model_file_name)

# 寻找与失望最接近的词
for k in model.similar_by_word('失望'):
    print(k[0],k[1])


# 创建词语字典
def create_dictionaries(p_model):
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(p_model.wv.index2word, allow_update=True)
    w2indx = {v: k  for k, v in gensim_dict.items()}  # 词语的索引，从0开始编号
    id2vec = {w2indx.get(word): model.wv.__getitem__(word) for word in w2indx.keys()}  # 词语的词向量
    return w2indx, id2vec

word_id_dic, id_vect_dic= create_dictionaries(model) # 两个词典的功能：word-> id , id -> vector
print('失望对应的id为：',word_id_dic['失望'])
print('失望对应的词向量vector为：',id_vect_dic[word_id_dic['失望']])


# token化数据，word->id
def get_tokenized_imdb(data):
    """
    data: list of [list of word , label]

    """
    for word_list, label in data:
        temp = []
        for word in word_list:
            if (word in word_id_dic.keys()):
                temp.append(int(word_id_dic[word]))
            else:
                temp.append(0)
        yield [temp, label]


# 对数据进行 截断 和 填充
def preprocess_imdb(data):
    max_l = 30  # 将每条微博通过截断或者补1，使得长度变成30

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [1] * (max_l - len(x))

    features = torch.tensor([pad(content[0]) for content in data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

data_train = preprocess_imdb(list(get_tokenized_imdb(data_seg_train)))
data_test = preprocess_imdb(list(get_tokenized_imdb(data_seg_test)))

# 加载数据到迭代器，并规定batch 大小
batch_size = 64
train_set = Data.TensorDataset(*data_train)  # *表示接受元组类型数组
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_set = Data.TensorDataset(*data_test)  # *表示接受元组类型数组
test_iter = Data.DataLoader(train_set, batch_size, shuffle=True)


class BiRNN(nn.Module):
    def __init__(self, vocab_num, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_num, embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens, 6)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs

vocab_num = len(model.wv.index2word)
embed_size, num_hiddens, num_layers = 30, 60, 2
net = BiRNN(vocab_num, embed_size, num_hiddens, num_layers)


id_vect = torch.Tensor(list(id_vect_dic.values()))
net.embedding.weight.data.copy_(id_vect)
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它


lr, num_epochs = 0.01, 50
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练函数
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        train_ls.append(train_l_sum / n)
        test_acc = evaluate_accuracy(test_iter, net)
        if ((epoch + 1) % 5 == 0):
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    loss_plot(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', ['train', 'test'])



# 评价函数
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n



# 定义绘图函数
def loss_plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)



train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)
