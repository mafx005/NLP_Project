# -*- coding: utf-8 -*-
# @Time : 2021/4/25 17:26
# @Author : M
# @FileName: attention_dot.py
# @Dec : 


from keras.regularizers import L1L2
import tensorflow as tf
from keras.layers import Layer, InputSpec, Flatten
from keras import backend as K
from keras import initializers


class AttentionDot(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='Attention_Dot_Weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 regularizer=L1L2(0.0000032),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Dot_Bias',
                                 regularizer=L1L2(0.00032),
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, input):
        x_transpose = K.permute_dimensions(input, (0, 2, 1))
        x_tanh_softmax = K.softmax(K.tanh(K.dot(x_transpose, self.W) + self.b))
        outputs = K.permute_dimensions(x_tanh_softmax * x_transpose, (0, 2, 1))
        # outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


class SelfAttention(Layer):
    """
        self attention,
        codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W、K and V
        self.kernel = self.add_weight(name='WKV',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      regularizer=L1L2(0.0000032),
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        # print("WQ.shape",WQ.shape)
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        # print("QK.shape",QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class CVG_Layer(Layer):
    def __init__(self, embed_size, filter, label, **kwargs):
        self.embed_size = embed_size
        self.filter = filter
        self.label = label
        super().__init__(**kwargs)

    def build(self, input_shape):
        self._filter = self.add_weight(name=f'filter_{self.filter}',
                                       shape=(self.filter, self.label, 1, 1),
                                       regularizer=L1L2(0.00032),
                                       initializer='uniform',
                                       trainable=True)
        self.class_w = self.add_weight(name='class_w',
                                       shape=(self.label, self.embed_size),
                                       regularizer=L1L2(0.0000032),
                                       initializer='uniform',
                                       trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(1,),
                                 regularizer=L1L2(0.00032),
                                 initializer='uniform',
                                 trainable=True)
        super().build(input_shape)

    def call(self, input):
        # C * V / G
        # l2_normalize of x, y
        input_norm = tf.nn.l2_normalize(input)  # b * s * e
        class_w_relu = tf.nn.relu(self.class_w)  # c * e
        label_embedding_reshape = tf.transpose(class_w_relu, [1, 0])  # e * c
        label_embedding_reshape_norm = tf.nn.l2_normalize(label_embedding_reshape)  # e * c
        # C * V
        G = tf.contrib.keras.backend.dot(input_norm, label_embedding_reshape_norm)  # b * s * c
        G_transpose = tf.transpose(G, [0, 2, 1])  # b * c * s
        G_expand = tf.expand_dims(G_transpose, axis=-1)  # b * c * s * 1
        # text_cnn
        conv = tf.nn.conv2d(name='conv', input=G_expand, filter=self._filter,
                            strides=[1, 1, 1, 1], padding='SAME')
        pool = tf.nn.relu(name='relu', features=tf.nn.bias_add(conv, self.b))  # b * c * s * 1
        # pool = tf.nn.max_pool(name='pool', value=h, ksize=[1, int((self.filters[0]-1)/2), 1, 1],
        #                       strides=[1, 1, 1, 1], padding='SAME')
        # max_pool
        pool_squeeze = tf.squeeze(pool, axis=-1)  # b * c * s
        pool_squeeze_transpose = tf.transpose(pool_squeeze, [0, 2, 1])  # b * s * c
        G_max_squeeze = tf.reduce_max(input_tensor=pool_squeeze_transpose, axis=-1, keepdims=True)  # b * s * 1
        # divide of softmax
        exp_logits = tf.exp(G_max_squeeze)
        exp_logits_sum = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
        att_v_max = tf.div(exp_logits, exp_logits_sum)
        # β * V
        x_att = tf.multiply(input, att_v_max)
        x_att_sum = tf.reduce_sum(x_att, axis=1)
        return x_att_sum

    def compute_output_shape(self, input_shape):
        return None, K.int_shape(self.class_w)[1]


if __name__ == "__main__":
    att = AttentionDot()
    self_att = SelfAttention(300)
