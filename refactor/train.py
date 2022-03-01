import time
import torch
from model import BiRNN


def train(net, optimizer, loss_func, train_iter, dev_iter, compute_val, epoches, load_model_dir, save_model_dir):
    if load_model_dir:
        net.load_state_dict(torch.load(load_model_dir))
    for epoch in range(epoches):
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i, data in enumerate(train_iter):
            x = data.text  # input输入

            y = data.label.squeeze(0)  # label真实标签

            target = net(x)  # target预测值标签

            l = loss_func(target, y)  # 计算预测和真实值的损失

            optimizer.zero_grad()  # 优化器权重清零
            l.backward()  # 损失反向传播
            optimizer.step()  # 优化器更新
            train_l_sum += l.item()  # 损失值求和
            train_acc_sum += (target.argmax(dim=1) == y).sum().item()  # 预测值等于真实值个数

            n += y.shape[0]  # y.shape[0]批次数量
            train_acc = train_acc_sum / n  # 预测等于真实值的正确个数除以总数得到正确率

            batch_count += 1
            if (i + 1) % 10 == 0:  # 每十个批次打印一次现在准确率
                print("Train accuracy now is %.1f" % (round(train_acc, 3) * 100) + '%')  # 现在的训练精度，保留三位小数百分比化
                if compute_val:
                    net.eval()  # 一种行为方式具体待理解，能固定参数不改变
                    val_data = next(iter(dev_iter))  # 读取验证集
                    dev_X = val_data.text  # 验证集输入
                    dev_y = val_data.label.squeeze(0)  # 验证集真实标签
                    val_score = net(dev_X)  # 验证集预测值
                    val_acc = (val_score.argmax(dim=1) == dev_y).sum().item() / len(dev_y)  # 验证集准确率
                    print("Val accuracy of one batch is %.1f " % (round(val_acc, 3) * 100) + '%')
                    net.train()  # 一种行为方式具体待理解
                print('*' * 25)
        if (epoch + 1) % 5 == 0 and True:
            print(f'saving model into => {save_model_dir}')
            torch.save(net.state_dict(),
                       save_model_dir)
            # torch.save()保存模型，torch.save(net,'net.pt')保存整个模型，torch.save(net.state_dict(),'net.pt')只保存训练好的权重
        print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc, time.time() - start))
        print(batch_count)
        # 打印轮数，当前轮数每个批次平均总损失，当前轮数现在准确率，当前轮数耗时


if __name__ == '__main__':

    print('end')
