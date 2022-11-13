from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity



# 如下是超参数设置
# 最大词数
MAX_SIZE = 10000
# 训练的词向量维度
embedding_size = 100
# 单边窗口数
single_win_size = 3
# 最大词频
x_max = 100
batch_size = 32
lr = 1e-3
epoch = 5


with open("data/test.txt", 'r', encoding='utf-8') as f:
    content = f.read().split(" ")

words = dict(Counter(content).most_common(MAX_SIZE-1))
words['<UNK>'] = len(content) - np.sum(list(words.values()))

word2idx = {word:i for i, word in enumerate(words.keys())}
idx2word = {i:word for i, word in enumerate(words.keys())}


def get_co_occurrence_matrix(content, word2idx):
    '''
    构建共现矩阵
    :param content: 文章分词后的列表
    :param word2idx: 字典（词：ID）
    :return: 共现矩阵
    '''
    # 初始化共现矩阵
    matrix = np.zeros((MAX_SIZE, MAX_SIZE), np.int32)
    # 单词列表转为编码
    content_encode = [word2idx.get(w, MAX_SIZE - 1) for w in content]
    # 遍历每一个中心词
    for i, center_id in enumerate(content_encode):
        # 取得同一窗口词在文中的索引
        pos_indices = list(range(i - single_win_size, i)) + list(range(i + 1, i + single_win_size + 1))
        # 取得同一窗口的词索引，避免越界，使用取模操作
        window = [j % len(content) for j in pos_indices]
        # 取得词对应的ID
        window_id = [content_encode[j] for j in window]
        # 使得中心词对应的背景词次数+1
        for j in window_id:
            matrix[center_id][j] += 1

    return matrix


matrix = get_co_occurrence_matrix(content, word2idx)


# # 得到惩罚矩阵，耗时太长，不采用这种方法
# def get_punish(matrix):
#     punish = np.zeros_like(matrix, np.float32)
#     # 因为共现矩阵是对称的，所以可以只遍历下三角
#     for i in range(MAX_SIZE):
#         for j in range(i+1):
#             punish[i][j] = matrix[i][j] ** (0.75) if matrix[i][j] < x_max else 1
#             punish[j][i] = punish[i][j]
#         if i % 1000 == 0:
#             print("第{}行".format(i))
#
# punish = get_punish(matrix)
# print('得到惩罚')


def get_nozero(matrix):
    index_nozero = []
    for i in range(MAX_SIZE):
        for j in range(i+1):
            if matrix[i][j] != 0:
                index_nozero.append([i,j])
                index_nozero.append([j,i])
    return index_nozero


index_nozero = get_nozero(matrix)
print("得到序列")

class GloVeDataset(Dataset):
    def __init__(self, matrix, index_nozero):
        super(GloVeDataset, self).__init__()  # 第一行必须是这个
        self.matrix = torch.Tensor(matrix)
        self.index_nozero = index_nozero


    def __len__(self):
        return len(index_nozero)


    def __getitem__(self, idx):
        row = self.index_nozero[idx][0]
        column = self.index_nozero[idx][1]
        x_ik = torch.tensor([self.matrix[row][column]])
        punish_x = torch.tensor([x_ik ** (0.75) if x_ik < x_max else 1])
        # x_ik = self.matrix[row][column]
        # punish_x = x_ik ** (0.75) if x_ik < x_max else 1

        return row, column, x_ik, punish_x


glove_dataset = GloVeDataset(matrix, index_nozero)
dataloader = DataLoader(glove_dataset, batch_size, shuffle=True)


class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(GloVe, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        # 中心词矩阵
        self.center_embed = nn.Embedding(self.vocab_size, self.embed_size)
        # 背景词矩阵
        self.backgroud_embed = nn.Embedding(self.vocab_size, self.embed_size)

        # 中心词偏置，偏置为一个常数，故为1维
        self.center_bias = nn.Embedding(self.vocab_size, 1)
        # 背景词偏置
        self.backgroud_bias = nn.Embedding(self.vocab_size, 1)
        # 随机初始化参数(这种初始化方式收敛更快)，embedding原来是默认（0,1）正态分布
        initrange = 0.5 / self.vocab_size
        self.center_embed.weight.data.uniform_(-initrange, initrange)
        self.backgroud_embed.weight.data.uniform_(-initrange, initrange)
        self.center_bias.weight.data.uniform_(-initrange, initrange)
        self.backgroud_bias.weight.data.uniform_(-initrange, initrange)



    def forward(self, row, column, x_ik, punish_x):
        '''
        注意输入是按批次输入的，所以其维度与批次一样
        :param row: [batch_size]
        :param column: [batch_size]
        :param x_ik: [batch_size, 1]
        :param punish_x: [batch_size, 1]
        :return:
        '''
        v_i = self.center_embed(row) # [batch_size, embed_size]
        u_k = self.backgroud_embed(column) # [batch_size, embed_size]
        b_i = self.center_bias(row)  # [batch_size, 1]
        # 需要将其变为一维才能正常做加法
        b_i = b_i.squeeze(1)  #[batch_size]

        b_k = self.backgroud_bias(column)  # [batch_size, 1]
        b_k = b_k.squeeze(1)

        x_ik = x_ik.squeeze(1)
        punish_x = punish_x.squeeze(1)

        # 按照损失函数计算损失即可
        loss = punish_x * (torch.mul(v_i, u_k).sum(dim=1) + b_i + b_k - torch.log(x_ik)) ** 2

        return loss


    def get_predic_vec(self):
        # 采用作者的方法，返回两者相加
        return self.center_embed.weight.data.cpu().numpy()+self.backgroud_embed.weight.data.cpu().numpy()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GloVe(MAX_SIZE, embedding_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train_model():
    #训练模型
    for e in range(epoch):
        for i, (row, clolumn, x_il, punish_x) in enumerate(dataloader):
            row = row.to(device)
            clolumn = clolumn.to(device)
            x_il = x_il.to(device)
            punish_x = punish_x.to(device)

            optimizer.zero_grad()
            loss = model(row, clolumn, x_il, punish_x).mean()
            loss.backward()

            optimizer.step()

            if i % 1000 == 0:
                print('epoch', e, 'iteration', i, loss.item())

    torch.save(model.state_dict(), "data/glove-{}.th".format(embedding_size))


train_model()

def find_word(word):
    '''
    计算并输出与输入词最相关的100个词
    :param word: 输入词
    :return:
    '''
    # 加载模型
    model = GloVe(MAX_SIZE, embedding_size)
    model.load_state_dict(torch.load("data/glove-100.th"))
    # 获取中心词矩阵
    embedding_weight = model.get_predic_vec()
    # 得到词与词向量的字典
    word2embedding = {}
    for i in words:
        word2embedding[i] = embedding_weight[word2idx[i]]
    # 得到输入词与其他词向量的余弦相似度
    other = {}
    for i in words:
        if i == word:
            continue
        # 计算余弦相似度
        other[i] = cosine_similarity(word2embedding[word].reshape(1, -1), word2embedding[i].reshape(1, -1))

    # 对余弦相似度按从大到小排序
    other = sorted(other.items(), key=lambda x: x[1], reverse=True)
    count = 0
    # 输出排序前100的相似度词语
    for i, j in other:
        print("({},{})".format(i, j))
        count += 1
        if count == 100:
            break


find_word('大师')


