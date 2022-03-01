from datasets import text_field, label_field
import pandas as pd
# import pymysql
import jieba
import torch

# from sklearn.model_selection import train_test_split

# 加载
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
def chi_tokenizer(sentence):
    # for word in jieba.cut(sentence):
    #     print(word)
    # print([word for word in jieba.cut(sentence)])
    return [word for word in jieba.cut(sentence)]


def transform_data(record, TEXT, LABEL):
    # [[2,8,9]] => [[7], [8], [9]]
    if not isinstance(record, dict):
        raise ValueError('Make sure data is dict')
    tokens = chi_tokenizer(record['text'])  # 测试集文本分词
    # print(tokens)
    res = []
    for token in tokens:
        res.append(TEXT.vocab.stoi[token])  # 分词对应词表向量化
    # print(res)
    # 向量列表转换为tensor张量，unsqueeze 下标为1的维度变成1，从0开始,squeeze删除对应下标维度为了1的维度
    data = torch.tensor(res).unsqueeze(1)
    # print(data.shape)
    # print(record['data'],record['label'],type(record['label']),'\n',list(record))
    # label = torch.tensor(LABEL.vocab.stoi['1'])
    # print(label)
    if 'label' in list(record):
        label = torch.tensor(LABEL.vocab.stoi[str(record['label'])])
        # print(label)
    else:
        label = None
    return data, label


if __name__ == '__main__':
    record_test = {'text': '很糟糕', 'label': 0}
    text, labels = transform_data(record_test, text_field, label_field)
    print(text, labels)
