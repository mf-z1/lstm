import torch
from torch import nn
from datasets import text_field, vocab_size

num_hiddens = 100
num_layers = 2
"""

torch.nn.Embedding： 随机初始化词向量，词向量值在正态分布N(0,1)中随机取值。

输入：
torch.nn.Embedding(
num_embeddings, – 词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999）
embedding_dim,– 嵌入向量的维度，即用多少维来表示一个符号。
padding_idx=None,– 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
max_norm=None, – 最大范数，如果嵌入向量的范数超过了这个界限，就要进行再归一化。
norm_type=2.0, – 指定利用什么范数计算，并用于对比max_norm，默认为2范数。
scale_grad_by_freq=False, 根据单词在mini-batch中出现的频率，对梯度进行放缩。默认为False.
sparse=False, – 若为True,则与权重矩阵相关的梯度转变为稀疏张量。
_weight=None)
输出：
[规整后的句子长度，样本个数（batch_size）,词向量维度]
"""


class BiRNN(nn.Module):

    def __init__(self, num_hiddens=num_hiddens, num_layers=num_layers):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size[0], vocab_size[1])  # 初始化词向量
        self.embedding.weight.data.copy_(text_field.vocab.vectors)  # 加载预训练词向量参数
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=vocab_size[1],  # 词向量维度
                               hidden_size=num_hiddens,  # 隐藏层维度
                               num_layers=num_layers,  # 用几层
                               bidirectional=True)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs)
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(encoding)
        return outs
