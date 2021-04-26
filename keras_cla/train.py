# -*- coding: utf-8 -*-
# @Time : 2020/11/30 15:45
# @Author : M
# @FileName: train.py
# @Dec : 

import os
import numpy as np
import random as rn
import tensorflow as tf
import keras
from gensim.models import Word2Vec
import gensim
import csv
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from Config.config import Config
from callbacks.My_Callback import Mylosscallback
from utils.util import load_data, load_vocab, predict2both, predict2tag


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = Config()

np.random.seed(42)
rn.seed(666)
tf.set_random_seed(2020)

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if config.do_train:
    try:
        os.remove(config.epoch_info_path)
    except:
        pass

train_text, train_label = load_data(config.train_path, config.class_file, config.cla_type, config.cla_nums, config.word_char)
valid_text, valid_label = load_data(config.valid_path, config.class_file, config.cla_type, config.cla_nums, config.word_char)
test_text, test_label = load_data(config.test_path, config.class_file, config.cla_type, config.cla_nums, config.word_char)
total_text = train_text + valid_text + test_text
print(len(total_text))
# print(train_label.min(), train_label.max())
train_text = train_text[:500]
valid_text = valid_text[:50]
test_text = test_text[:5]
print(test_text)
train_label = train_label[:500]
valid_label = valid_label[:50]
test_label = test_label[:5]
print(test_label)

if config.word_char == 'word':
    token = load_vocab(config.word_vocab_path, total_text, word_char='word', num_words=config.word_num_words)
else:
    token = load_vocab(config.char_vocab_path, total_text, word_char='char', num_words=config.char_num_words)

if config.word_char == 'word':
    word_vocab = token.word_index
    if not os.path.exists(config.w2v_path):
        print("++++++++++train word2vec model finished++++++++++")
        model = Word2Vec([[word for word in document.split(' ')] for document in total_text],
                         size=300,
                         window=5,
                         iter=10,
                         seed=2020,
                         min_count=2)
        model.save(config.w2v_path)
        w2v_model = Word2Vec.load(config.w2v_path)
        print("++++++++++train word2vec model finished++++++++++")
    else:
        w2v_model = Word2Vec.load(config.w2v_path)
        # glove_model = gensim.models.KeyedVectors.load_word2vec_format(config.glove_path)
        print("++++++++++load word2vec glove finished++++++++++")

    w2v_embedding_matrix = np.zeros((len(word_vocab) + 1, 300))
    for word, i in word_vocab.items():
        embedding_vector = w2v_model.wv[word] if word in w2v_model else None
        if embedding_vector is not None:
            w2v_embedding_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(300) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            w2v_embedding_matrix[i] = unk_vec
    # glove_embedding_matrix = np.zeros((len(word_vocab) + 1, 300))
    # for word, i in word_vocab.items():
    #     embedding_vector = glove_model.wv[word] if word in glove_model else None
    #     if embedding_vector is not None:
    #         glove_embedding_matrix[i] = embedding_vector
    #     else:
    #         unk_vec = np.random.random(300) * 0.5
    #         unk_vec = unk_vec - unk_vec.mean()
    #         glove_embedding_matrix[i] = unk_vec
    if config.embedding_type == 'glove' and os.path.exists(config.glove_path):
        # embedding_matrix = glove_embedding_matrix
        embedding_matrix = w2v_embedding_matrix
        print('glove embedding load finish')
    elif config.embedding_type == 'w2v_glove':
        # embedding_matrix = np.concatenate((w2v_embedding_matrix, glove_embedding_matrix),axis=1)
        embedding_matrix = w2v_embedding_matrix
        print('w2v & glove embedding load finish')
    else:
        embedding_matrix = w2v_embedding_matrix
        print('word2vec embedding load finish')
    model = config.model(config.word_maxlen, embedding_matrix, config.word_char)
else:
    model = config.model(config.char_maxlen, None, config.word_char)

if config.do_train:
    print('++++++++++Training++++++++++')
    train = token.texts_to_sequences(train_text)
    valid = token.texts_to_sequences(valid_text)
    if config.word_char == 'word':
        train = keras.preprocessing.sequence.pad_sequences(train, maxlen=config.word_maxlen)
        valid = keras.preprocessing.sequence.pad_sequences(valid, maxlen=config.word_maxlen)
    else:
        train = keras.preprocessing.sequence.pad_sequences(train, maxlen=config.char_maxlen)
        valid = keras.preprocessing.sequence.pad_sequences(valid, maxlen=config.char_maxlen)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=1, mode='max', factor=0.5, patience=3)
    checkpoint_prefix = os.path.join(config.checkpoint_dir, "best.weight")
    checkpoint = ModelCheckpoint(checkpoint_prefix, monitor='val_loss', verbose=2,
                                 save_best_only=True, mode='max', save_weights_only=True)
    if config.cla_type == 'Multiclass':
        print('softmax training')
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    if config.cla_type in ['Binary_class', 'Mutillabel']:
        print('sigmoid training')
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    model.fit(train, train_label, batch_size=config.batch_size, epochs=config.epoch,
              validation_data=(valid, valid_label),
              callbacks=[early_stopping, plateau, checkpoint, Mylosscallback(log_dir='log')],
              verbose=1, shuffle=False)
    model.summary()
    # plot_model(model, to_file=os.path.join(config.checkpoint_dir, 'model.png'), show_shapes=True)

if config.do_predict:
    print('++++++++++Predicting++++++++++')
    test = token.texts_to_sequences(test_text)
    if config.word_char == 'word':
        test = keras.preprocessing.sequence.pad_sequences(test, maxlen=config.word_maxlen)
    else:
        test = keras.preprocessing.sequence.pad_sequences(test, maxlen=config.char_maxlen)

    model.load_weights(os.path.join(config.checkpoint_dir, 'best.weight'))
    print('Model_load_path:', os.path.join(config.checkpoint_dir, 'best.weight'))
    if config.cla_type == 'Multiclass':
        test_pred = model.predict(test)
        pred_labels = np.argmax(test_pred, axis=1)
        # pred_labels = list(map(str, pred_labels))
        test_label = np.argmax(test_label, axis=1)
    elif config.cla_type == 'Binary_class':
        test_pred = model.predict(test)
        pred_labels = np.where(test_pred > 0.8, 1, 0)
    elif config.cla_type == 'Mutillabel':
        test_pred = model.predict(test)
        pred_labels = predict2both(test_pred)
        pred_labels_tag = predict2tag(pred_labels, config.class_file)
        test_labels_tag = predict2tag(test_label, config.class_file)
        tmp = [(test_label[i] == pred_labels[i]).min() for i in range(len(pred_labels))]
        test_accu = sum(tmp) / len(tmp)
        f1_micro = f1_score(y_pred=pred_labels, y_true=test_label, pos_label=1, average='micro')
        f1_macro = f1_score(y_pred=pred_labels, y_true=test_label, pos_label=1, average='macro')
        f1_avg = (f1_micro + f1_macro) / 2
    else:
        raise Exception('Assert cla_type in Multiclass or Binary_class or Mutillabel')
    if config.cla_type in ['Multiclass', 'Binary_class']:
        test_acc = accuracy_score(test_label, pred_labels)
        test_f1 = f1_score(test_label, pred_labels, average='macro')
        test_recall = recall_score(test_label, pred_labels, average='macro')
        test_precision = precision_score(test_label, pred_labels, average='macro')

        print('ACC:{}'.format(test_acc))
        print('F1:{}'.format(test_f1))
        print('recall:{}'.format(test_recall))
        print('precision:{}'.format(test_precision))
        print(confusion_matrix(test_label, pred_labels))
        print(classification_report(test_label, pred_labels))
    elif config.cla_type in ['Mutillabel']:
        test_acc = test_accu
        test_f1 = f1_avg
        test_recall = ''
        test_precision = ''
        f1_macro = f1_macro
        f1_micro = f1_micro
    else:
        raise Exception('Assert cla_type in Multiclass or Binary_class or Mutillabel')
    # 写入最终结果
    if config.write2txt:
        print('++++++++++Write Result ++++++++++')
        with open(config.result_info_path, 'w', encoding='utf8')as f:
            f.write('ACC:' + '\t' + str(test_acc) + '\n')
            f.write('F1:' + '\t' + str(test_f1) + '\n')
            f.write('recall:' + '\t' + str(test_recall) + '\n')
            f.write('precision:' + '\t' + str(test_precision) + '\n')
        f.close()

    # bad case输出
    if config.write_badcase:
        print('badcase写入中')
        pred_labels_list = pred_labels.tolist()
        true_labels_list = test_label.tolist()
        bad_case_file = open(config.badcase_path, 'w', encoding='utf8', newline="")
        csv_writer = csv.writer(bad_case_file)
        csv_writer.writerow(['fact', 'true_label', 'pred_label', 'pred_probability'])
        test_file = open(config.test_path, 'r', encoding='utf8')
        test_lines = test_file.readlines()
        # print(test_lines[0])
        if config.cla_type == 'Mutillabel':
            for i in range(len(tmp)):
                if not tmp[i]:
                    csv_writer.writerow([test_lines[i].split('\t')[1], test_labels_tag[i], pred_labels_tag[i], ''])
        else:
            for i in range(len(pred_labels_list)):
                if pred_labels_list[i][0] != true_labels_list[i]:
                    # print(i)
                    csv_writer.writerow([test_lines[i].split('\t')[1], test_lines[i].split('\t')[0], pred_labels_list[i][0], test_pred.tolist()[i][0]])
        bad_case_file.close()
