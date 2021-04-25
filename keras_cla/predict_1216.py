# coding=utf-8
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from classification_model import BiGRU, TextCNN
from gensim.models import Word2Vec
import pickle
import jieba
import numpy as np
import json
from timeit import default_timer as timer
from config import Config
import gensim

config = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def load_w2v_model(token):
    w2v_model = Word2Vec.load(config.w2v_path)
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(config.glove_path)
    word_vocab = token.word_index
    w2v_embedding_matrix = np.zeros((len(word_vocab) + 1, 300))
    for word, i in word_vocab.items():
        embedding_vector = w2v_model.wv[word] if word in w2v_model else None
        if embedding_vector is not None:
            w2v_embedding_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(300) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            w2v_embedding_matrix[i] = unk_vec
    glove_embedding_matrix = np.zeros((len(word_vocab) + 1, 300))
    for word, i in word_vocab.items():
        embedding_vector = glove_model.wv[word] if word in glove_model else None
        if embedding_vector is not None:
            glove_embedding_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(300) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            glove_embedding_matrix[i] = unk_vec
    embedding_matrix = np.concatenate((w2v_embedding_matrix, glove_embedding_matrix),axis=1)
    print('embedding init finish')
    return embedding_matrix


class Predictor(object):
    def __init__(self):
        self.token = pickle.load(open(config.word_vocab_path, 'rb'))
        print('load vocab finish')
        self.w2v_embedding_matrix = load_w2v_model(self.token)
        self.gru_model = BiGRU(config.word_maxlen, self.w2v_embedding_matrix, config.word_char)
        self.model = self.gru_model.load_weights(tf.train.latest_checkpoint(config.checkpoint_dir))
        print('load model finish')

    def fact_predict(self, text_list):
        test_text = []
        pred_fact = []
        pred_res = []
        text_list = text_list.split('\n')
        for i in range(len(text_list)):
            tmp = text_list[i]
            tmp = tmp.replace(' ', '')
            if config.word_char == 'word':
                test_text.append(list(jieba.cut(tmp)))
            else:
                test_text.append([x for x in tmp])
            # print(test_text)
            test_text_sequence = self.token.texts_to_sequences(test_text)
            test_text = sequence.pad_sequences(test_text_sequence, maxlen=config.word_maxlen)
            # result = self.gru_model.predict(test_text)
            
            if config.cla_type == 'Multiclass':
                test_pred = self.gru_model.predict(test_text)
                pred_labels = np.argmax(test_pred, axis=1)
                pred_fact.append(str(pred_labels[0]) + '\t' + tmp)
            if config.cla_type == 'Binary_class':
                test_pred = self.gru_model.predict(test_text)
                if result[0][0] > 0.8:
                    pred_fact.append(tmp)

                result[result >= 0.8] = 1
                result[result < 0.8] = 0
                pred_res.append(int(result[0][0]))
            test_text = []
        pred_fact = list(filter(None, pred_fact))
        return text_list, pred_fact, pred_res


# if __name__ == '__main__':
    # start_time_1 = timer()
    # str_test = '''
     # 四黎明，四川路石律师事务所律师。
# 
# 　　人民陪审员王少松
# 　　人民陪审员刘抒
# 　　二Ｏ一一年四月二十九日
# 　　书记员廖莉


    # '''
    # # start_time1 = timer()
    # # ajjbqk = data_preprocess(str_test)
    # # _, res, _ = pre.fact_predict(ajjbqk)
    # # print(res)
    # start_time = timer()
    # pre = Predictor()
    # dir_name = r'test_data/3、盗窃罪'
    # out_name = r'test_data/'

    # for maindir, subdir, file_name_list in os.walk(dir_name):
        # for j in file_name_list:
            # with open(os.path.join(dir_name, j), 'r', encoding='utf8')as f:
                # text_string = f.read()
                # ajjbqk = data_preprocess(text_string)
                # _, res, _ = pre.fact_predict(ajjbqk)
                # if not res:
                    # print('=================')
                    # print(j)
                    # print('=================')
                # res = merge_fact_paragraphs(res)
                # res = json.dumps(res, ensure_ascii=False)
            # with open(os.path.join(out_name, j.split('.')[0] + '.json'), "w", encoding='utf8') as fo:
                # fo.write(res)
# end_time1 = timer()
# print(end_time1 - start_time_1)


pre = Predictor()
out_file = open('temp/model_predict_znfz.txt', 'w', encoding='utf8')
with open('data/智能辅助_起诉意见书_段落id.txt', 'r', encoding='utf8')as f:
    line = f.readline()
    count = 0
    while line:
        count += 1
        # if count == 5:
            # break
        if count % 1000 == 0:
            print(count)
        _, res, _ = pre.fact_predict(line)
        res = res[0]
        out_file.write(res + '\n')
        line = f.readline()
out_file.close()
        