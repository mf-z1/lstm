from torch import optim
from torch import nn
from train import BiRNN
from train import train
from datasets import train_iter, dev_iter

lr = 0.005  # 学习率

# 获取模型名称
net = BiRNN()  # 选择模型

optimizer = optim.Adam(net.parameters(), lr=lr)  # 随机梯度优化器
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失
compute_val = 'store_true'  # 执行验证集
epoches = 5  # 训练轮数
save_model_dir = 'models_storage/model_lstm.pt'  # 模型保存路径
load_model_dir = ''  # 加载模型参数路径

if __name__ == '__main__':
    train(net=net, optimizer=optimizer, loss_func=loss_func,
          train_iter=train_iter, dev_iter=dev_iter,
          compute_val=compute_val, epoches=epoches,
          load_model_dir=load_model_dir, save_model_dir=save_model_dir)
