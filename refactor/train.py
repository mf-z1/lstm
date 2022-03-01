import time
import torch
from model import BiRNN


# from datasets import train_iter, dev_iter

# net = BiRNN()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 随机梯度优化器
# loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失
# epoches = 5  # 训练轮数
# save_model_dir = 'models_storage/model_lstm.pt'
# load_model_dir = ''


def train(net, optimizer, loss_func, train_iter, dev_iter, compute_val, epoches, load_model_dir, save_model_dir):
    # batch_count = 0
    # pass
    if load_model_dir:
        net.load_state_dict(torch.load(load_model_dir))
    for epoch in range(epoches):
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i, data in enumerate(train_iter):
            x = data.text  # input输入
            # print(x, x.shape)
            y = data.label.squeeze(0)  # label真实标签
            # print(y, y.shape, y.shape[0])
            target = net(x)  # target预测值标签
            # print(target, target.shape)
            l = loss_func(target, y)  # 计算预测和真实值的损失
            # print('-------', l, l.item())
            optimizer.zero_grad()  # 优化器权重清零
            l.backward()  # 损失反向传播
            optimizer.step()  # 优化器更新
            train_l_sum += l.item()  # 损失值求和
            train_acc_sum += (target.argmax(dim=1) == y).sum().item()  # 预测值等于真实值个数
            # print('******', (target.argmax(dim=1) == y).sum().item())
            n += y.shape[0]  # y.shape[0]批次数量
            train_acc = train_acc_sum / n  # 预测等于真实值的正确个数除以总数得到正确率
            # print(train_acc)
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
    # from datasets import train_iter, dev_iter
    #
    # net = BiRNN()
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 随机梯度优化器
    # loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失
    # epoches = 5  # 训练轮数
    # save_model_dir = 'models_storage/model_lstm.pt'
    # load_model_dir = ''
    # compute_val = 'compute_val'
    # train(net, optimizer, loss_func, train_iter, dev_iter, compute_val, epoches, load_model_dir, save_model_dir)
    print('end')
# def train(net, optimizer, loss_func, train_iter, dev_iter, compute_val,
#           device, epoches, load_model_dir, save_model_dir):
#     '''
#
#     :param net:
#     :param optimizer:
#     :param loss_func:
#     :param train_iter:
#     :param val_iter:
#     :param compute_val:
#     :param device:
#     :param epoches:
#     :param load_model_dir:
#     :param save_model_dir:
#     :return:
#     '''
#     print(f'>>>We are gonna tranning {net.__class__.__name__} with epoches of {epoches}<<<')
#     net = net.to(device)
#     if load_model_dir:
#         net.load_state_dict(torch.load(load_model_dir))
#     batch_count = 0
#     for epoch in range(epoches):
#         print(f'=>we are training epoch[{epoch + 1}]...<=')
#         train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
#         for iter_num, batch in enumerate(train_iter):
#             X = batch.text.to(device)
#             print(X, X.shape)
#             y = batch.label.squeeze(0).to(device)
#             score = net(X)
#             print(score)
#             try:
#                 l = loss_func(score, y)  # 一定是score在前， y在后！
#             except:
#                 print('error occured!!')
#                 print(f'score shape:{score.shape}; y shape:{y.shape}; y:{y}')
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (score.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#             train_acc = train_acc_sum / n
#             if (iter_num + 1) % 10 == 0:
#                 print("Train accuracy now is %.1f" % (round(train_acc, 3) * 100) + '%')
#                 ## 计算validation score
#                 if compute_val:
#                     net.eval()
#                     val_data = next(iter(dev_iter))
#                     dev_X = val_data.data.to(device)
#                     dev_y = val_data.label.squeeze(0).to(device)
#                     val_score = net(dev_X)
#                     val_acc = (val_score.argmax(dim=1) == dev_y).sum().cpu().item() / len(dev_y)
#                     print("Val accuracy of one batch is %.1f " % (round(val_acc, 3) * 100) + '%')
#                     net.train()
#                 print('*' * 25)
#         if (epoch + 1) % 5 == 0 and save_model_dir:
#             print(f'saving model into => {save_model_dir}')
#             torch.save(net.state_dict(), save_model_dir)
#         print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
#               % (epoch + 1, train_l_sum / batch_count, train_acc, time.time() - start))
