import re
import torchtext.legacy.data as data
import jieba
import logging
from torchtext.vocab import Vectors
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Example

jieba.setLogLevel(logging.INFO)

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
jieba.load_userdict('.././dictionary/机构_学校.lex')

embedding_loc = '.././vocab/cc.zh.300.vec'  # 本地词向量路径
cache = '.vector_cache'  # 词向量的缓存位置


def word_cut(text):
    text = regex.sub(' ', text)
    # print(text)
    return [word for word in jieba.cut(text) if word.strip()]


text_field = data.Field(lower=True)
label_field = data.Field(sequential=False, eos_token=None, pad_token=None, unk_token=None)
text_field.tokenize = word_cut  # 分词方法
# print(text_field.tokenize)
train, dev = data.TabularDataset.splits(
    path='.././data', format='tsv', skip_header=True,
    train='train.tsv', validation='dev.tsv',
    fields=[
        ('index', None),
        ('label', label_field),
        ('text', text_field)
    ]
)
# print(vars(dev.examples[0]))


vectors = Vectors(name=embedding_loc, cache=cache)  # 构建词向量表

text_field.build_vocab(train, dev, vectors=vectors)  # 向量化
label_field.build_vocab(train, dev, vectors=vectors)  # 向量化
# 训练数据集
train_iter = BucketIterator(train, batch_size=64,
                            sort_key=lambda x: len(x.text), sort_within_batch=True, shuffle=True)
# 验证数据
dev_iter = BucketIterator(dev, batch_size=64,
                          sort_key=lambda x: len(x.text), sort_within_batch=True, shuffle=True)

vocab_size = text_field.vocab.vectors.shape  # 词表维度
# print(vocab_size)
if __name__ == '__main__':
    print(word_cut('我的时间2.0t不多了'))
