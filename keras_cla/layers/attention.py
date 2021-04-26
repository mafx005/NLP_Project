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


if __name__ == "__main__":
    att = AttentionDot()
    self_att = SelfAttention(300)
