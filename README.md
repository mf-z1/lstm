#  lstm情感二分类实验

该项目是使用pytorch的基于BiLST双向长短记忆神经网络的文本分类实验：对汽车相关的短文本进行情感二分类。该项目也可以扩展为多分类问题。

## 项目的整体结构

```                
├── data                   原始数据
├── dictionary             用户自定义词典
├── vocab                  词向量
└── refactor               函数体
    ├── templates          api输入界面
    ├── datasets.py        数据格式转换
    ├── test.py            模型在测试集上测试效果
    ├── main.py            主程序
    ├── model.py           文本模型
    ├── models_storage     模型保存目录
    ├── train.py           训练相关函数
    ├── api.py             用于预测的api程序
    └── utils.py           工具函数
```

### 训练模型

（1） 准备好数据 data目录下准备好训练数据集：train.tsv、dev.tsv...数据要保证有以下字段：text(文本字段)、 label(分类标签， 该例分为0和1， 0代表消极， 1代表积极)。

（2） 下载词向量  
从网上下载预训练好的词向量， 比如FastText词向量, 你可从[该处](https://fasttext.cc/docs/en/crawl-vectors.html)
下载一个300维的中文词向量， 然后将解压出来的txt词向量文本文件放在vocab/目录下（在配置文件标明相应的词向量地址）

（3）运行

### api调用

api调用结果图(figs/input.png show.png)



