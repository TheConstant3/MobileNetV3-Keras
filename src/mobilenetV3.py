from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


'''
    from paper https://arxiv.org/pdf/1905.02244.pdf:
        exp size ~ filters
        #out ~ out (amount filters on last conv layer)
        SE ~ has_se
        NL ~ activation
        s ~ scale
'''


class MobileNetV3Small():
    def __init__(self, n_classes):
        self.input_shape = (224, 224, 3)
        self.n_classes = n_classes

    def bneck(self, input_x, kernel=3, filters=16, out=16,
              has_se=False, activation=None, s=1):
        x = Conv2D(filters, kernel_size=1, strides=1)(input_x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        x = DepthwiseConv2D(kernel_size=kernel, strides=s, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        if has_se:
            x = self.se_block(x)

        x = Conv2D(out, kernel_size=1, strides=1)(x)

        if input_x.shape[-1] == x.shape[-1] and s == 1:
            return Add()([x, input_x])

        return x

    def se_block(self, input_x):
        num_chanels = int(input_x.shape[-1])
        x = GlobalAveragePooling2D()(input_x)
        x = Dense(num_chanels//4, activation='relu')(x)
        x = Dense(num_chanels, activation='hard_sigmoid')(x)
        return Multiply()([input_x, x])

    def relu6(self, x):
        return K.relu(x, max_value=6.0)

    def hard_swish(self, x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def biuld_model(self):
        x = input_x = Input(shape=self.input_shape)

        x = Conv2D(16, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(self.hard_swish)(x)

        x = self.bneck(x, kernel=3, filters=16, out=16,
                       has_se=True, activation=self.relu6, s=2)

        x = self.bneck(x, kernel=3, filters=72, out=24,
                       has_se=False, activation=self.relu6, s=2)
        x = self.bneck(x, kernel=3, filters=88, out=24,
                       has_se=False, activation=self.relu6, s=1)

        x = self.bneck(x, kernel=5, filters=96, out=40,
                       has_se=True, activation=self.hard_swish, s=2)
        x = self.bneck(x, kernel=5, filters=240, out=40,
                       has_se=True, activation=self.hard_swish, s=1)
        x = self.bneck(x, kernel=5, filters=240, out=40,
                       has_se=True, activation=self.hard_swish, s=1)

        x = self.bneck(x, kernel=5, filters=120, out=48,
                       has_se=True, activation=self.hard_swish, s=1)
        x = self.bneck(x, kernel=5, filters=144, out=48,
                       has_se=True, activation=self.hard_swish, s=1)

        x = self.bneck(x, kernel=5, filters=288, out=96,
                       has_se=True, activation=self.hard_swish, s=2)
        x = self.bneck(x, kernel=5, filters=576, out=96,
                       has_se=True, activation=self.hard_swish, s=1)
        x = self.bneck(x, kernel=5, filters=576, out=96,
                       has_se=True, activation=self.hard_swish, s=1)

        x = Conv2D(576, kernel_size=1, strides=1)(x)
        x = BatchNormalization()(x)
        x = Activation(self.hard_swish)(x)

        x = self.se_block(x)

        x = AveragePooling2D(pool_size=7, strides=1)(x)

        x = Conv2D(1024, kernel_size=1, strides=1)(x)
        x = Activation(self.hard_swish)(x)

        out = Conv2D(self.n_classes, kernel_size=1, strides=1)(x)

        out = Reshape((10,))(out)

        return Model(input_x, out)
