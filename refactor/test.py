import pandas as pd
import torch
from utils import transform_data
from model import BiRNN
from datasets import text_field, label_field

load_model_dir = 'models_storage/model_lstm.pt'  # 模型参数保存路径
# 获取模型名称
net = BiRNN()  # 选择模型
net.load_state_dict(torch.load(load_model_dir))  # 加载模型参数


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
    evaluate(model=net, df=test_data)
