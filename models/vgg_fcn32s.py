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

    x = encorder.outputs[0]

    # dimention reduction
    x = Conv2D(classes, 1, activation='relu', name='conv_p7',
               kernel_regularizer=l2(weight_decay),
               kernel_initializer='he_normal')(x)

    # upsampling x32
    if bilinear:
        x = BilinearUpSampling2D((32, 32))(x)
    else:
        x = Conv2DTranspose(classes, 64, activation='relu',
                            strides=32, padding='same', name='upscore_final',
                            kernel_regularizer=l2(weight_decay),
                            #kernel_initializer=Constant(bilinear_upsample_weights(8, classes)),
                            kernel_initializer='he_normal')(x)

    x = Activation('softmax', name='softmax')(x)

    model = Model(encorder_input, x)

    return model
