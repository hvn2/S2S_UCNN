import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, BatchNormalization,Conv3D
from tensorflow.keras.layers import LeakyReLU,Concatenate, Add
from models.common import *
""""Define SkipNet structure used in the paper
    Iput:   ndown: number of convolution blocks (K in the paper)
            channel: output channel
    Ouput:  model
    """
def skip(ndown=5,channelin=10,channelout=6):
    #down side
    input_layer = Input((None,None,channelin))
    out = input_layer
    skips=[]

    for i in range(ndown):
        out = convbn(out,dilation_rate=1)
        skips.append(convbn(out,filters=25,kernel_size=1))
    skips.reverse()
    for i in range(ndown):
        if i==0:
            out=convbn(out)
        else:
            out = convbn(Concatenate()([out,skips[i]]))

    out = Conv2D(channelout,1)(out)
    # out=convbn(out,filters=channelout,kernel_size=1)
    out_layer = Add()([input_layer[:, :, :, 0:6], out])

    mymodel=Model(input_layer,out_layer)
    return mymodel



def residualnet(K=24,channelin=10,channelout=6):
    input_layer = Input((None,None,channelin))
    out = input_layer
    out=Conv2D(64,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    first_out=out
    for i in range(K):
        out=resblock(out)

    out=Conv2D(64,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    out=Conv2D(64,3,padding="same")(out)
    out=Add()([first_out,out])
    out=Conv2D(128,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    out=Conv2D(256,3,padding="same")(out)
    out=LeakyReLU(alpha=0.01)(out)
    out=Conv2D(filters=channelout,kernel_size=3,padding="same")(out)
    out_layer=Add()([input_layer[:,:,:,0:6],out])
    # out_layer=Conv2D(channelout,1)(out_layer)

    model=Model(input_layer,out_layer)
    return model
