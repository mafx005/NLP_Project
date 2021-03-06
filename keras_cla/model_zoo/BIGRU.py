# -*- coding: utf-8 -*-
# @Time : 2021/4/26 9:51
# @Author : M
# @FileName: bigru.py
# @Dec :

from keras.layers import Embedding, Bidirectional, GRU, GlobalAveragePooling1D, BatchNormalization
from keras.layers import Activation, concatenate, Dropout, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D, Dense, Input
from keras.models import Model


def BiGRUModel(sent_length, embeddings_weight, word_char):
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
        embedding = Embedding(name='char_embedding', input_dim=config.char_num_words, output_dim=config.embed_size, trainable=True)
    x = SpatialDropout1D(0.2)(embedding(content))
    x = Bidirectional(GRU(400, return_sequences=True))(x)
    x = Bidirectional(GRU(400, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(1024)(conc)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
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
