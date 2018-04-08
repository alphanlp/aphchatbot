# -*- coding:utf-8 -*-

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import data_utils

"""
实现一个基于tf-idf的简易检索系统   
实现一个基于lsi / plsa的简易检索系统
"""
corpus = [
    "尊敬的领导您好，我是加州水郡Q酷的业主，2016年通过中介买了商用房，面积39.77，有房本性质是酒店，因为想要换房，所以今年3月初找到买家到房山信建委想办理网签业务，结果被告知不能办网签，因为我一直没看到有政策规定酒店性质的房子不能买卖，所以极其苦恼，如果不一直不能交易，那我当初花了40万买的房，通过交易大厅正规有房本的房子，难道就不能交易了吗？急求答疑解惑，非常感谢！！！！",
    "本人户籍所在地湖南省长沙市，在北京工作，打算在长沙市买房，想用北京的公积金贷款，长沙公积金部门说要在缴存公积金所在地开无房证明，求问在哪儿可以开？",
    "您好，我长年在北京上班，在外省购房，办理公积金贷款。现在需要工作地开具”房屋产权证明“，证明下自己在北京有无购房情况，请问是在这办理吗",
    "密码错误，找回时收不到手机验证码",
    "住建委的各位领导：您好！2018年1~2月公示的共有产权房申请，第一批是针对京籍，网上下拉菜单只有“北京籍”选项。且于3月9日公示了初步审核结果。请问针对非京籍的网上在线申请，何时开始？是否留有30%的份额针对非京籍？谢谢！",
    "共有产权房属于保障房吗？",
    "看新闻说住公租房能迁户口应该怎样办理，请回复。谢谢",
    "请问住房公积金最高贷款为120W，首付为20%，这个规定看起来没问题，但是实际应用在北京市是有些自相矛盾的，将来是否会通过一些政策来为在北京工作的刚需购房者提供一些便利，比如适当提升利率提高可贷款额度",
    "我想请问一下，我是外国人，在北京工作，有工作签证和正常住房公积金。我在北京买期房，是否只能买有外销证的楼盘？如果是，我如何查询有外销证的楼盘有哪些？相关流程和规定我在哪里可以查到？是否可以买任意二手房？首付贷款比例等要求是否与公民一样？购买和售出时，是否有相比中国公民以外的，新增的税费？多谢",
    "本人是京户，老公是外籍。申请资料第7条，夫妇双方一方户口不在申请所在地需提供其户口所在地的住房证明；他在外地也没有住房，这个证明怎么开？申请资料第8条/第9条/第10条/第11条，需要家庭成员分别出具收入证明/社保对账单/公积金缴存个人信息/完税证明，他已经快2年没有工作了，一直也没有上社保公积金，这些资料如何开具，是否还需要提供？"
]


class TfidfBot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b', stop_words=["，", ",", "啊", "的"])

    def train(self, corpus):
        self.corpus = [" ".join(jieba.cut(line, cut_all=False)) for line in corpus]
        self.context = self.vectorizer.fit_transform(self.corpus)
        # words = self.vectorizer.get_feature_names()
        # for i in range(len(self.corpus)):
        #     print('----Document %d----' % (i))
        #     for j in range(len(words)):
        #         print(words[j], self.context[i, j])

    def predict(self, utterance):
        utterance = " ".join(jieba.cut(utterance, cut_all=False))
        utter_vec = self.vectorizer.transform([utterance])
        # print(utter_vec)
        # sub_corpus = self.corpus[3:7]
        # id_dict = {}
        # for i, value in enumerate(sub_corpus):
        #     id_dict[i] = 3 + i
        # self.context = self.vectorizer.transform(self.corpus)
        # print("\n")
        # print(self.context)
        result = np.dot(utter_vec, self.context.T).todense()
        # print(result)
        result = np.asarray(result, dtype=np.float32).flatten()
        sorted_result = np.argsort(result, axis=0)[::-1]
        # return np.array([id_dict[i] for i in sorted_result]), result
        return sorted_result, result

def tfidf_test():
    tfidfBot = TfidfBot()
    corpus = data_utils.load_tolist(path='./resources/question.txt')
    tfidfBot.train(corpus)
    selectId, result_value = tfidfBot.predict("海淀区公租房申请")
    # print top 3
    for i in range(3):
        print("%f   %s" % (result_value[selectId[i]], corpus[selectId[i]]))
    # print(selectId)
    # print(result_value)


def jieba_test():
    text = "你知道谁么"
    text_cut = jieba.cut(text, cut_all=False)
    for word in text_cut:
        print(word)


if __name__ == '__main__':
    # jieba_test()
    tfidf_test()