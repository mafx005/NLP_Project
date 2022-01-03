# -*- coding: utf-8 -*-
# @Time    : 2022/1/3 16:19
# @Author  : M.
# @File    : tfidf.py
# @Describe: tfidf计算文本相似度

import jieba
import numpy as np

# corpus = [
#     '我在北京天安门',
#     '选择AI，就是选择未来',
#     '要么996要么icu',
#     '我爱加班，加班使我快乐'
# ]
#
# filepath = '../data/hit_stopwords.txt'
# stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#
# word_list = []
# for cor in corpus:
#     seg_list = jieba.cut(cor)
#     seg_list = [i for i in seg_list if i not in stopwords and i!=' ']
#     word_list.append(seg_list)
# print(word_list)


class tf_idf(object):
    def __init__(self, docs_list):
        # docs_list中的文本需要先分词
        self.docs_list = docs_list
        self.docs_len = len(docs_list)
        self.tf = []
        self.idf = {}
        self.init()

    def init(self):
        df = {}
        for doc in self.docs_list:
            temp = {}
            for word in doc:
                temp[word] = temp.get(word, 0) + 1/len(doc)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log(self.docs_len / (value + 1))

    def get_score(self, index, query):
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_docs_score(self, query):
        score_list = []
        for i in range(self.docs_len):
            score_list.append(self.get_score(i, query))
        return score_list

if __name__ == '__main__':
    document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
                     "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
                     "我在微信上被骗了，请问被骗多少钱才可以立案？",
                     "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
                     "有人走私两万元，怎么处置他？",
                     "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]
    document_list = [list(jieba.cut(doc)) for doc in document_list]
    print(document_list)
    tf_idf_model = tf_idf(document_list)
    print(tf_idf_model.docs_list)
    print(tf_idf_model.docs_len)
    print(tf_idf_model.tf)
    print(tf_idf_model.idf)
    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))
    scores = tf_idf_model.get_docs_score(query)

    print(scores)
