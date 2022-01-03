# -*- coding: utf-8 -*-
# @Time    : 2022/1/3 16:52
# @Author  : M.
# @File    : bm25.py
# @Describe: bm25计算文本相似度

import numpy as np
import jieba
from collections import Counter


class BM25_sim(object):
    def __init__(self, docs_list, k1=1, k2=1, b=0.5):
        self.docs_list = docs_list
        self.docs_len = len(docs_list)
        self.avg_docs_len = sum([len(doc) for doc in docs_list]) / self.docs_len
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for doc in self.docs_list:
            temp = {}
            for word in doc:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.docs_len - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        doc_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * doc_len / self.avg_docs_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

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
    # exit()
    bm25_model = BM25_sim(document_list)
    print(bm25_model.docs_list)
    print(bm25_model.docs_len)
    print(bm25_model.avg_docs_len)
    print(bm25_model.f)
    print(bm25_model.idf)
    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))
    scores = bm25_model.get_docs_score(query)
    print(scores)