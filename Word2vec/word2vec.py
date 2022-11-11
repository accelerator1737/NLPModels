import re
import jieba
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from gensim.models import Word2Vec
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


MAX_SIZE = 10000
# 训练的词向量维度
embedding_size = 100
C = 3 # context window
K = 15 # number of negative samples
batch_size = 32
lr = 1e-3
epoch = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_file(file):
    stopwords = {}
    with open('data/Stop.txt', 'r', encoding='utf-8', errors='ingnore') as f:
        for eachWord in f:
            stopwords[eachWord.strip()] = eachWord.strip()  # 创建停用词典

    with open('data/{}'.format(file), 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Unicode的\u4e00-\u9fa5为中文，用正则表达式将中文抽出，达到去除标点符号的作用
    content = re.findall('[\u4e00-\u9fa5]+', content)

    concentrate = []
    # 遍历每一个句子，对每一个句子进行分词，效果更好
    for i in content:
        seg_sentence = jieba.cut(i, cut_all=False)
        concentrate.extend([j for j in seg_sentence])

    main_content = []

    # 去除停用词
    for i in concentrate:
        if i not in stopwords:
            main_content.append(i)

    # 存储词的列表，词与词用空格隔开
    with open("data/test.txt", 'w', encoding='utf-8') as f:
        f.write(' '.join(main_content))

# make_file('douluo-utf.txt')

with open("data/test.txt", 'r', encoding='utf-8') as f:
    content = f.read().split(" ")

words = dict(Counter(content).most_common(MAX_SIZE-1))
words['<UNK>'] = len(content) - np.sum(list(words.values()))

word2idx = {word:i for i, word in enumerate(words.keys())}
idx2word = {i:word for i, word in enumerate(words.keys())}

word_counts = np.array([count for count in words.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)


class WordEmbeddingDataset(Dataset):
    def __init__(self, text, word2idx, word_freqs):
        ''' text: a list of words, all text from the training dataset
            word2idx: the dictionary from word to index
            word_freqs: the frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__()  # #通过父类初始化模型，然后重写两个方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]  # 把单词数字化表示（27）。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded)  # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)  # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        center_words = self.text_encoded[idx]  # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]  # 为了避免索引越界，所以进行取余处理
        pos_words = self.text_encoded[pos_indices]  # tensor(list)背景词表所引

        # multinomial表示对self.word_freqs按照概率大小抽取K * pos_words.shape[0]个样本，True表示是放回抽取
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        # while 循环是为了保证 neg_words中不能包含背景词
        while len(set(np.array(pos_indices).tolist()) & set(np.array(neg_words).tolist())) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_words, pos_words, neg_words


# print(len(content))
words_dataset = WordEmbeddingDataset(content, word2idx, word_freqs)
dataloader = DataLoader(words_dataset, batch_size, shuffle=True)

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        # 计算正样本损失
        log_pos = F.logsigmoid(pos_dot).sum(1) # [batch_size]
        # 计算负样本损失
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg

        return -loss

    def get_embedding(self):
        return self.in_embed.weight.cpu().detach().numpy()


model = EmbeddingModel(MAX_SIZE, embedding_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train_model():
    #训练模型
    for e in range(epoch):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long().to(device)
            pos_labels = pos_labels.long().to(device)
            neg_labels = neg_labels.long().to(device)

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()

            optimizer.step()

            if i % 1000 == 0:
                print('epoch', e, 'iteration', i, loss.item())

    torch.save(model.state_dict(), "data/embedding-{}.th".format(embedding_size))


# train_model()

def find_word(word):
    '''
    计算并输出与输入词最相关的100个词
    :param word: 输入词
    :return:
    '''
    model = EmbeddingModel(MAX_SIZE, embedding_size)
    model.load_state_dict(torch.load("data/embedding-100.th"))
    embedding_weight = model.get_embedding()
    word2embedding = {}
    for i in words:
        word2embedding[i] = embedding_weight[word2idx[i]]

    other = {}

    for i in words:
        if i == word:
            continue
        other[i] = cosine_similarity(word2embedding[word].reshape(1,-1), word2embedding[i].reshape(1,-1))
    other = sorted(other.items(), key=lambda x: x[1], reverse=True)
    count = 0
    for i, j in other:
        print("({},{})".format(i,j))
        count += 1
        if count == 100:
            break

find_word('大师')