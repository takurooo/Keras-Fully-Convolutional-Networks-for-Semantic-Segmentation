#-----------------------------------------
# import
#-----------------------------------------
import os
import argparse
from keras.applications import VGG16
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation, MaxPooling2D, Dropout, UpSampling2D
from keras.models import Model
from keras.initializers import Constant
from keras.regularizers import l2
import numpy as np
from models.layers import BilinearUpSampling2D, bilinear_upsample_weights
#-----------------------------------------
# defines
#-----------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))
#-----------------------------------------
# functions
#-----------------------------------------


def get_args():
    parser = argparse.ArgumentParser(description="models.")
    return parser.parse_args()


def build(classes=21, input_shape=(224, 224, 3), drop_rate=None, bilinear=False):

    vgg16 = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

    '''
    Base Encorder
    '''
    input = Input(shape=input_shape)

    x = vgg16.layers[1](input)
    x = vgg16.layers[2](x)
    x = vgg16.layers[3](x)
    x = vgg16.layers[4](x)
    x = vgg16.layers[5](x)
    x = vgg16.layers[6](x)
    x = vgg16.layers[7](x)
    x = vgg16.layers[8](x)
    x = vgg16.layers[9](x)
    x = vgg16.layers[10](x)
    # block3_pool
    p3 = x
    x = vgg16.layers[11](x)
    x = vgg16.layers[12](x)
    x = vgg16.layers[13](x)
    x = vgg16.layers[14](x)
    # block4_pool
    p4 = x
    x = vgg16.layers[15](x)
    x = vgg16.layers[16](x)
    x = vgg16.layers[17](x)
    x = vgg16.layers[18](x)

    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc6')(x)
    if drop_rate:
        x = Dropout(drop_rate)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc7')(x)
    if drop_rate:
        x = Dropout(drop_rate)(x)
    p7 = x

    '''
    Skip Connection
    '''
    # dimention reduction
    p3 = Conv2D(classes, 1, activation='relu', name='conv_p3')(p3)
    p4 = Conv2D(classes, 1, activation='relu', name='conv_p4')(p4)
    p7 = Conv2D(classes, 1, activation='relu', name='conv_p7')(p7)

    # upsampling x2
    u4 = Conv2DTranspose(
        classes, 4, activation='relu',
        #kernel_initializer=Constant(bilinear_upsample_weights(2, classes)),
        strides=2, padding='same', name='upscore_p4')(p4)
    # upsampling x4
    u7 = Conv2DTranspose(
        classes, 8, activation='relu',
        #kernel_initializer=Constant(bilinear_upsample_weights(4, classes)),
        strides=4, padding='same', name='upscore_p7')(p7)

    x = Add(name='add')([p3, u4, u7])

    # upsampling x8
    if bilinear:
        x = BilinearUpSampling2D((8, 8))(x)
    else:
        x = Conv2DTranspose(
            classes, 16, activation='relu',
            #kernel_initializer=Constant(bilinear_upsample_weights(8, classes)),
            strides=8, padding='same', name='upscore_final')(x)

    x = Activation('softmax', name='softmax')(x)

    return Model(input, x)


#-----------------------------------------
# main
#-----------------------------------------


def main(args):
    model = build()
    print(model.summary())


if __name__ == '__main__':
    print("start")
    main(get_args())
    print("end")
