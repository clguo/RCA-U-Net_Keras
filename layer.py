
from keras.layers import *


def reduce_mean(input):
    return K.mean(input, axis=(1,2), keepdims=True)
def RCAB(input, reduction):
    channel= input.get_shape().as_list()[-1]
    f = Conv2D(channel, (3,3), padding='same', activation='relu')(input)  # (B, W, H, C)
    f = Conv2D(channel, (3,3), padding='same')(f) # (B, W, H, C)
    x=Lambda(reduce_mean)(f)
    x = Conv2D(channel // reduction, (1,1), activation='relu')(x)  # (B,,1, 1, C // r )
    x = Conv2D(channel, (1,1), activation='sigmoid')(x)
    x = multiply([f,x])  # (B, c, w ,h)
    x = add([input,x])
    return x