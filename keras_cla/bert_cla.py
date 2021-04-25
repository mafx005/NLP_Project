#! -*- coding:utf-8 -*-
# 情感分析例子，加载albert_zh权重(https://github.com/brightmart/albert_zh)

import numpy as np
import os
import tensorflow as tf
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

set_gelu('tanh')  # 切换gelu版本

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

num_classes = 1
maxlen = 200
batch_size = 16
epoch = 50
config_path = r'PTM_path/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = r'PTM_path/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'PTM_path/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            label, text = l.strip().split('\t')
            if len(text) > maxlen:
                D.append(((text.strip()[:maxlen]), int(label.strip())))
            else:
                D.append((text.strip(), int(label.strip())))
    return D


# 加载数据集
train_data = load_data('data/word/origin_dataset/qss_pjs_augment_shuffle.txt')
valid_data = load_data('data/word/origin_dataset/valid_qss_pjs_augment_shuffle.txt')
test_data = load_data('data/word/origin_dataset/2020_gjcs_data_word.txt')


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='binary_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-5, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    y_pred_list = []
    y_true_list = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        y_pred = np.where(y_pred > 0.8, 1, 0)
        y_pred = y_pred[:, 0]
        # print(y_pred)
        y_true = y_true[:, 0]
        y_pred_list += y_pred.tolist()
        y_true_list += y_true.tolist()
        total += len(y_true)
        right += (y_true == y_pred).sum()
    test_acc = accuracy_score(y_true_list, y_pred_list)
    test_f1 = f1_score(y_true_list, y_pred_list, average='macro')
    test_recall = recall_score(y_true_list, y_pred_list, average='macro')
    test_precision = precision_score(y_true_list, y_pred_list, average='macro')
    print(confusion_matrix(y_true_list, y_pred_list))
    return right / total, test_acc, test_f1, test_recall, test_precision


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc, sk_val_acc, sk_val_f1, sk_val_recall, sk_val_precision = evaluate(valid_generator)
        # print('Val acc',val_acc)
        # print('sk_val_acc',sk_val_acc)
        # print('sk_val_f1',sk_val_f1)
        # print('sk_val_recall',sk_val_recall)
        # print('sk_val_precision',sk_val_precision)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('save_model/bert_model/best_model.weights')
        test_acc, sk_test_acc, sk_test_f1, sk_test_recall, sk_test_precision = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f, sk_test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc, sk_test_acc)
        )


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        callbacks=[evaluator]
    )

    model.load_weights('save_model/bert_model/best_model.weights')
    print(u'final test acc: %05f final test sk_acc: %05f final test sk_f1: %05f final test sk_recall: %05f final test sk_precision: %05f\n' % (evaluate(test_generator)))

else:

    model.load_weights('save_model/bert_model/best_model.weights')
