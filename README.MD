# 项目介绍

BERT-TEXT-PRETTY 是一个高效的BERT模型文本编码解码器，可快速对文本进行各类的编码与解码，
可用于BERT模型的训练、预测、部署推理等场景。

# 安装

'''
pip install bert-text-pretty
'''

# 使用说明

目前可以对分类模型,序列模型,关系抽取模型的文本进行编码与解码。


## 分类模型编码与解码

```python

# -*- coding:utf-8 -*-
import numpy as np
from bert_text_pretty import cls, ner, relation, tokenization

# 文本数据
text_list = ["你是谁123456", "你是谁123456222222222222"]

# 指定BERT模型字典，生成tokenizer对象
tokenizer = tokenization.FullTokenizer(vocab_file=r'F:\models\chinese_L-12_H-768_A-12\vocab.txt',
                                        do_lower_case=True)
# 分类模型编码
feat = cls.cls_text_feature(tokenizer, text_list, max_seq_len=128, with_padding=False)
print(feat)
'''
(array([[  101,   872,  3221,  6443, 10644,   102,     0,     0,     0,
            0,     0,     0],
       [  101,   872,  3221,  6443, 10644,  8683,  8683,  8683,  8683,
         8683,  8683,   102]]), array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]))
'''

```

### 解码

```

# 标签字典
labels = ['标签1','标签2', '标签3']
# 加载字典，也可以从字典文件中加载，字典文件为TXT格式，一行一个标签；
id2label = cls.load_labels(labels)
print(id2label)

# 样本数据
example_all = ['分类样本句子1', '分类样本句子2']
# 模型输出结果
logits = np.array([[0.10532469, 0.18736999, 0.19014127],
       [0.22059808, 0.16185541, 0.13471056]])
# 分类文本解码
ret = cls.cls_decoding(example_all, id2label, logits)
'''
   example_all 文本list ,
   id2label 标签列表或者字典 list or dict
   logits_all (batch,hidden)
   threshold 阈值
'''

print(ret)
'''
输出结果：
[['标签3'], ['标签1']]

'''
```


## NER编码与解码样例

```python

# -*- coding:utf-8 -*-
import numpy as np
from bert_text_pretty import cls,ner,relation,tokenization


# 文本数据
text_list = ["你是谁123456","你是谁123456222222222222"]

# 指定BERT模型字典，生成tokenizer对象
tokenizer = tokenization.FullTokenizer(vocab_file=r'F:\models\chinese_L-12_H-768_A-12\vocab.txt',
                                        do_lower_case=True)


# 序列模型编码
feat = ner.ner_text_feature(tokenizer,text_list,max_seq_len=128,with_padding=False)
print(feat)

# 关系模型编码
feat = relation.re_text_feature(tokenizer,text_list,max_seq_len=128,with_padding=False)
print(feat)

# 解码
labels = ['标签1','标签2']
print(ner.load_label_bio(labels))

# def ner_decoding(example_all, id2label, logits_all,trans=None)
'''
    example_all 文本list ,
    id2label 标签 list or dict
    logits_all 为bert 预测结果 (batch,seq_len,num_tags) or (batch,seq_len)
    trans 是否启用trans预测 , 2D
    解析crf序列  or 解析 已经解析过的crf序列

'''


#ner_pointer_decoding(example_all, id2label, logits_all,threshold=0.):
'''
   example_all 文本list ,
   id2label 标签 list or dict
   logits_all (batch,num_labels,seq_len,seq_len)
   threshold 阈值

'''






# relation.re_decoding(example_all, id2spo, logits_all)  #关旭


```