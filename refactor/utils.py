from datasets import text_field, label_field
import jieba
import torch

# 加载
jieba.load_userdict('.././dictionary/机构_学校.lex')


# 定义一个tokenizer
def chi_tokenizer(sentence):
    # for word in jieba.cut(sentence):
    #     print(word)
    # print([word for word in jieba.cut(sentence)])
    return [word for word in jieba.cut(sentence)]


def transform_data(record, TEXT, LABEL):
    if not isinstance(record, dict):
        raise ValueError('Make sure data is dict')
    tokens = chi_tokenizer(record['text'])  # 测试集文本分词

    res = []
    for token in tokens:
        res.append(TEXT.vocab.stoi[token])  # 分词对应词表向量化

    # 向量列表转换为tensor张量，unsqueeze 下标为1的维度变成1，从0开始,squeeze删除对应下标维度为了1的维度
    data = torch.tensor(res).unsqueeze(1)

    if 'label' in list(record):
        label = torch.tensor(LABEL.vocab.stoi[str(record['label'])])
    else:
        label = None
    return data, label


if __name__ == '__main__':
    record_test = {'text': '很糟糕', 'label': 0}
    text, labels = transform_data(record_test, text_field, label_field)
    print(text, labels)
