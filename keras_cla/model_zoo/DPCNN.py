# -*- coding: utf-8 -*-
# @Time : 2021/4/26 9:55
# @Author : M
# @FileName: DPCNN.py
# @Dec : 


from keras.layers import Conv1D
from keras.layers import Embedding, BatchNormalization
from keras.layers import Dropout, GlobalMaxPooling1D
from keras.layers import Dense, Input, PReLU
from keras.layers import add, MaxPooling1D
from keras.models import Model
from keras.regularizers import l2

filter_nr = 250
bias_reg = l2(0.00001)
kernel_reg = l2(0.00001)


def DPCNNModel(sent_length, embeddings_weight, word_char):
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
    x = embedding(content)
    x = BatchNormalization()(x)

    region_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg)(x)
    region_emb = PReLU()(region_emb)

    # block1 = Conv1D(filter_nr, kernel_size=3, padding='same', activation='linear',
    # kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(x)

    block1 = Conv1D(filter_nr, kernel_size=3, padding='same', activation='linear',
                    kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(region_emb)

    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=3, padding='same', activation='linear',
                    kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    block1_out = add([block1, region_emb])
    x = MaxPooling1D(pool_size=3, strides=2)(block1_out)

    for i in range(7):
        block = Conv1D(filter_nr, kernel_size=3, padding='same', activation='linear',
                       kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(x)
        block = BatchNormalization()(block)
        block = PReLU()(block)
        block = Conv1D(filter_nr, kernel_size=3, padding='same', activation='linear',
                       kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)(block)
        block = BatchNormalization()(block)
        block = PReLU()(block)
        block_out = add([x, block])
        if i + 1 != 7:
            x = MaxPooling1D(pool_size=3, strides=2)(block_out)
    x = GlobalMaxPooling1D()(block_out)
    output = Dense(512, activation='linear')(x)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(0.5)(output)
    if config.cla_type == 'Binary_class':
        output = Dense(1, activation="sigmoid")(output)
    elif config.cla_type == 'Multiclass':
        output = Dense(config.cla_nums, activation="softmax")(output)
    elif config.cla_type == 'Mutillabel':
        output = Dense(config.cla_nums, activation="sigmoid")(x)
    else:
        raise Exception('Assert cla_type in Multiclass or Binary_class or Mutillabel')

    model = Model(inputs=content, outputs=output)

    return model
