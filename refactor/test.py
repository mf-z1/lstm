'''
  Evaluate model performence
'''
import pandas as pd
import os
import torch
import argparse
# from uits import transform_data
from model import BiRNN
from datasets import text_field, label_field
import jieba

# 确保存在test.csv
# if not os.path.exists('data/test.csv'):
#     load_data_to_csv(flag='test')


# parser = argparse.ArgumentParser()
# parser.add_argument('--model-name', default='birnn', choices=['textcnn', 'birnn'],
#                     help='choose one model name for trainng')
# parser.add_argument('-lmd', '--load-model-dir', default='models_storage/model_brnn.pt',
#                     help='path for loadding model, default:None')
# args = parser.parse_args()
load_model_dir = 'models_storage/model_lstm.pt'  # 模型参数保存路径
# 获取模型名称
net = BiRNN()  # 选择模型
net.load_state_dict(torch.load(load_model_dir))  # 加载模型参数

jieba.load_userdict('.././dictionary/机构_学校.lex')


# def load_data_to_csv(flag):
#     # conn = pymysql.connect(**bc.db_connection)
#     if flag == 'train':
#         data = pd.read_sql('select data,label from weibo_sentiment where flag="train"')
#         train_data, val_data = train_test_split(data, test_size=0.1)
#         train_data.to_csv("data/train.csv", index=False)
#         val_data.to_csv("data/val.csv", index=False)
#     elif flag == 'test':
#         test_data = pd.read_sql('select data,label from weibo_sentiment where flag="test"')
#         test_data.to_csv('data/test.csv')
#     else:
#         raise ValueError('flag must be in ("test", "train")')


# 定义一个tokenizer
# 分词方法
# def chi_tokenizer(sentence):
#     # for word in jieba.cut(sentence):
#     #     print(word)
#     # print([word for word in jieba.cut(sentence)])
#     return [word for word in jieba.cut(sentence)]
#
#
# # 测试集向量化方法
# def transform_data(record, TEXT, LABEL):
#     # [[2,8,9]] => [[7], [8], [9]]
#     if not isinstance(record, dict):
#         raise ValueError('Make sure data is dict')
#     tokens = chi_tokenizer(record['text'])  # 测试集文本分词
#     # print(tokens)
#     res = []
#     for token in tokens:
#         res.append(TEXT.vocab.stoi[token])  # 分词对应词表向量化
#     # print(res)
#     data = torch.tensor(res).unsqueeze(1)  # 向量列表转换为tensor张量，下标为1的维度变成1，从0开始
#     # print(data.shape)
#     # print(record['data'],record['label'],type(record['label']),'\n',list(record))
#     # label = torch.tensor(LABEL.vocab.stoi['1'])
#     # print(label)
#     if 'label' in list(record):
#         label = torch.tensor(LABEL.vocab.stoi[str(record['label'])])
#         # print(label)
#     else:
#         label = None
#     return data, label


def evaluate(model, df):
    result = {'correct': 0, 'wrong': 0}
    df_len = df.shape[0]
    for i in range(df_len):
        record = df.loc[i, :].to_dict()
        print(record['text'])
        data, label = transform_data(record, text_field, label_field)
        score = model(data)
        if label == 0:
            true = '负向'
        else:
            true = '正向'
        if int(score.argmax(dim=1)) == 0:
            tge = '负向'
        else:
            tge = '正向'
        print('真实值:{},预测值:{}'.format(true, tge))
        if score.argmax(dim=1) == label:
            result['correct'] += 1
        else:
            result['wrong'] += 1
    print(f"Classification Accuracy of Model({model.__class__.__name__})is {result['correct'] / df_len} ")


if __name__ == '__main__':
    test_data = pd.read_csv('.././data/test.tsv', sep='\t')
    # print(test_data)
    evaluate(model=net, df=test_data)
