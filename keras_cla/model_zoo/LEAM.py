# -*- coding: utf-8 -*-
# @Time    : 2021/4/26 21:38
# @Author  : M.
# @Email   : mafx5218@126.com
# @File    : LEAM.py
# @Describe:


from keras.layers import Embedding
from keras.layers import Dropout, Concatenate
from keras.layers import Dense, Input
from keras.models import Model
from layers.attention import AttentionDot
from layers.leam_cvg_layer import CVG_Layer

filters = [3, 4, 5]


def LEAMModel(sent_length, embeddings_weight, word_char):
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
    x = AttentionDot()(x)
    pools = []
    for filter in filters:
        x_cvg = CVG_Layer(config.embed_size, filter, config.cla_nums)(x)
        pools.append(x_cvg)
    pools_concat = Concatenate()(pools)
    # MLP
    x_cvg_dense = Dense(int(config.embed_size / 2), activation="relu")(pools_concat)
    output = Dropout(0.5)(x_cvg_dense)
    if config.cla_type == 'Binary_class':
        output = Dense(1, activation="sigmoid")(output)
    elif config.cla_type == 'Multiclass':
        output = Dense(config.cla_nums, activation="softmax")(output)
    elif config.cla_type == 'Mutillabel':
        output = Dense(config.cla_nums, activation="sigmoid")(output)
    else:
        raise Exception('Assert cla_type in Multiclass or Binary_class or Mutillabel')

    model = Model(inputs=content, outputs=output)

    return model
