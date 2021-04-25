# -*- coding: utf-8 -*-
# @Time : 2020/11/30 16:04
# @Author : M
# @FileName: classification_model.py
# @Dec : 

from keras.layers import Conv1D
from keras.layers import Embedding, Bidirectional, GRU, GlobalAveragePooling1D, BatchNormalization
from keras.layers import Activation, concatenate, Dropout, GlobalMaxPooling1D, Concatenate
from keras.layers import SpatialDropout1D, Dense, Input, PReLU
from keras.layers import add, MaxPooling1D, CuDNNGRU, CuDNNLSTM, LSTM
from keras.models import Model
from keras.regularizers import l2
from layers_ import AttentionWeightedAverage


filter_nr = 250
bias_reg = l2(0.00001)
kernel_reg = l2(0.00001)


def TextCNN(sent_length, embeddings_weight, word_char):
    content = Input(shape=(sent_length,), dtype='int32')
    from config import Config
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
    x = Dropout(0.5)(x)
    convs = []
    for kernel_size in [2, 3, 4, 5]:
        c = Conv1D(512, kernel_size, activation='relu')(x)
        c = GlobalMaxPooling1D()(c)
        # c = KMaxPooling(k=3)(c)
        convs.append(c)
    x = Concatenate()(convs)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    if config.cla_type == 'Multiclass':
        output = Dense(config.cla_nums, activation="softmax")(x)
    if config.cla_type == 'Binary_class':
        output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=content, outputs=output)

    return model


def BiGRU(sent_length, embeddings_weight, word_char):
    content = Input(shape=(sent_length,), dtype='int32')
    from config import Config
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
    if config.cla_type == 'Multiclass':
        output = Dense(config.cla_nums, activation="softmax")(x)
    if config.cla_type == 'Binary_class':
        output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=content, outputs=output)
    return model


def DPCNN(sent_length, embeddings_weight):
    content = Input(shape=(sent_length,), dtype='int32')
    from config import Config
    config = Config()
    embedding = Embedding(
        name="word_embedding",
        input_dim=embeddings_weight.shape[0],
        weights=[embeddings_weight],
        output_dim=embeddings_weight.shape[1],
        trainable=True)
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
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=content, outputs=output)

    return model


def AttRNN(sent_length, embeddings_weight, word_char):
    content = Input(shape=(sent_length,), dtype='int32')
    from config import Config
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
    if config.cla_type == 'Multiclass':
        output = Dense(config.cla_nums, activation="softmax")(x)
    if config.cla_type == 'Binary_class':
        output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=content, outputs=output)
    return model
