# -*- coding: utf-8 -*-
# @Time : 2021/4/26 17:24
# @Author : M
# @FileName: han.py
# @Dec : 


from keras.layers import Dense, Dropout, Input, Embedding, Flatten
from keras.layers import Bidirectional, GRU
from keras import regularizers
from keras.models import Model
from layers.attention import SelfAttention


def HANModel(sent_length, embeddings_weight, word_char):
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
    x_word = word_level(sent_length, config.embed_size)(x)
    x_word_to_sen = Dropout(0.2)(x_word)
    x_sen = sentence_level(sent_length, config.embed_size)(x_word_to_sen)
    x = Dropout(0.2)(x_sen)
    x = Flatten()(x)
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


def word_level(sent_length, emb_zise):
    x_input_word = Input(shape=(sent_length, emb_zise))
    # x = SpatialDropout1D(0.2)(x_input_word)
    x = Bidirectional(GRU(units=256,
                          return_sequences=True,
                          activation='relu',
                          kernel_regularizer=regularizers.l2(1e-6),
                          recurrent_regularizer=regularizers.l2(1e-6)))(x_input_word)
    out_sent = SelfAttention(512)(x)
    model = Model(x_input_word, out_sent)
    return model


def sentence_level(sent_length, emb_zise):
    x_input_sen = Input(shape=(sent_length, 512))
    # x = SpatialDropout1D(self.dropout_spatial)(x_input_sen)
    output_doc = Bidirectional(GRU(units=512,
                                   return_sequences=True,
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(1e-6),
                                   recurrent_regularizer=regularizers.l2(1e-6)))(x_input_sen)
    output_doc_att = SelfAttention(emb_zise)(output_doc)
    model = Model(x_input_sen, output_doc_att)
    return model
