import torch
import os
import argparse
from torch import optim
from torch import nn
from train import BiRNN
from train import train
from datasets import train_iter, dev_iter

# from utils import load_data_to_csv
# from configs import  BasicConfigs


# 确保有'train.csv' 文件在指定目录下
# if not os.path.exists('data/train.csv'):
#     load_data_to_csv(flag='train')
#
# bc = BasicConfigs()
# 参数
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 使用gpu还是cpu
lr = 0.005  # 学习率

# 获取参数
# parser = argparse.ArgumentParser()
# parser.add_argument('--compute-val', action='store_true', help='compute validation accuracy or not, default:None')
# parser.add_argument('--epoches', default=5, type=int, help='num of epoches for trainning loop, default:20')
# parser.add_argument('-lmd', '--load-model-dir', default=None, help='path for loadding model, default:None')
# parser.add_argument('-smd', '--save-model-dir', default='models_storage/model_brnn.pt',
#                     help='models_storage/model_cnn.pt, defaul:None')
# parser.add_argument('--model-name', default='birnn', choices=['textcnn', 'birnn'],
#                     help='choose one model name for trainng')
# args = parser.parse_args()

# 获取模型名称
net = BiRNN()  # 选择模型
# device = device

optimizer = optim.Adam(net.parameters(), lr=lr)  # 随机梯度优化器
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失
compute_val = 'store_true'  # 执行验证集
epoches = 5  # 训练轮数
save_model_dir = 'models_storage/model_lstm.pt'  # 模型保存路径
load_model_dir = ''  # 加载模型参数路径

if __name__ == '__main__':
    # print(args.compute_val)

    train(net=net, optimizer=optimizer, loss_func=loss_func,
          train_iter=train_iter, dev_iter=dev_iter,
          compute_val=compute_val, epoches=epoches,
          load_model_dir=load_model_dir, save_model_dir=save_model_dir)
