#-----------------------------------------
# import
#-----------------------------------------
import os
import sys
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation, MaxPooling2D, Dropout, UpSampling2D
from keras.models import Model
from keras.initializers import Constant
from keras.regularizers import l2
from .layers import BilinearUpSampling2D, bilinear_upsample_weights
from .encorders import build_vgg16


#-----------------------------------------
# defines
#-----------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

#-----------------------------------------
# functions
#-----------------------------------------


def build(classes=21, input_shape=(224, 224, 3), weights_path=None, weight_decay=0., drop_rate=None, bilinear=False):

    # Build Base Encorder
    encorder = build_vgg16(input_shape, weights_path,
                           weight_decay, drop_rate)

    encorder_input = encorder.inputs[0]
    # for skip connection
    p3 = encorder.get_layer(name='block3_pool').output
    p4 = encorder.get_layer(name='block4_pool').output
    p7 = encorder.outputs[0]

    '''
    Skip Connection
    '''
    # dimention reduction
    p3 = Conv2D(classes, 1, activation='relu', name='conv_p3',
                kernel_regularizer=l2(weight_decay),
                kernel_initializer='he_normal')(p3)
    p4 = Conv2D(classes, 1, activation='relu', name='conv_p4',
                kernel_regularizer=l2(weight_decay),
                kernel_initializer='he_normal')(p4)
    p7 = Conv2D(classes, 1, activation='relu', name='conv_p7',
                kernel_regularizer=l2(weight_decay),
                kernel_initializer='he_normal')(p7)

    # upsampling x2
    u4 = Conv2DTranspose(classes, 4, activation='relu',
                         strides=2, padding='same', name='upscore_p4',
                         kernel_regularizer=l2(weight_decay),
                         #kernel_initializer=Constant(bilinear_upsample_weights(2, classes)),
                         kernel_initializer='he_normal')(p4)

    # upsampling x4
    u7 = Conv2DTranspose(classes, 8, activation='relu',
                         strides=4, padding='same', name='upscore_p7',
                         kernel_regularizer=l2(weight_decay),
                         #kernel_initializer=Constant(bilinear_upsample_weights(4, classes)),
                         kernel_initializer='he_normal')(p7)

    # fuse skip layers
    x = Add(name='add')([p3, u4, u7])

    # upsampling x8
    if bilinear:
        x = BilinearUpSampling2D((8, 8))(x)
    else:
        x = Conv2DTranspose(classes, 16, activation='relu',
                            strides=8, padding='same', name='upscore_final',
                            kernel_regularizer=l2(weight_decay),
                            #kernel_initializer=Constant(bilinear_upsample_weights(8, classes)),
                            kernel_initializer='he_normal')(x)

    x = Activation('softmax', name='softmax')(x)

    model = Model(encorder_input, x)

    return model
