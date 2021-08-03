import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D,LayerNormalization,BatchNormalization,LeakyReLU,Add


def convbn(x,filters=128,kernel_size=3,strides=1,padding="same",dilation_rate=1,kernel_initializer="he_normal",batchnorm=True):
    x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding,dilation_rate=dilation_rate, kernel_initializer=kernel_initializer)(x)
    # x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
    #            kernel_initializer=kernel_initializer)(x)
    if batchnorm:
        x=LayerNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    return x
#
# def resblock(x,filters=64,kernel_size=3,strides=1,padding="same",kernel_initializer="he_normal"):
#     x_shortcut=x
#     x=Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)
#     x = LeakyReLU()(x)
#     x=Conv2D(filters=filters,kernel_size=1,strides=strides,padding=padding)(x)
#     #x must be the same dimension of x_shortcut
#     x=Add()([x,x_shortcut])
#     return x