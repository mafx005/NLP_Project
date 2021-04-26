# -*- coding: utf-8 -*-
# @Time : 2021/4/26 9:57
# @Author : M
# @FileName: rnn_att.py
# @Dec : 


from keras.layers import Embedding, Bidirectional, BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.models import Model
from layers.attention import AttentionWeightedAverage


def AttRNN(sent_length, embeddings_weight, word_char):
    content = Input(shape=(sent_length,), dtype='int32')
    from Config.config import Config
    config = Config()
    if word_char == 'word':
        embedding = Embedding(
            name="word_embedding",
            input_dim=embeddings_weight.shape[0],
            weights=[embeddings_weight],
            output_dim=embeddings_weight.shape[1],
            trainable=True)
    else:
        print('执行基于字的训练模型')
        embedding = Embedding(name='char_embedding', input_dim=config.char_num_words, output_dim=config.embed_size,
                              trainable=True)

    x = embedding(content)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(LSTM(400, return_sequences=True, unit_forget_bias=False))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(400, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = AttentionWeightedAverage()(x)

    x = Dropout(0.5)(x)
    if config.cla_type == 'Binary_class':
        output = Dense(1, activation="sigmoid")(x)
    elif config.cla_type == 'Multiclass':
        output = Dense(config.cla_nums, activation="softmax")(x)
    elif config.cla_type == 'Mutillabel':
        output = Dense(config.cla_nums, activation="sigmoid")(x)
    else:
        raise Exception('Assert cla_type in Multiclass or Binary_class or Mutillabel')
    model = Model(inputs=content, outputs=output)
    return model
