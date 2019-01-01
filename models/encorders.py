#-----------------------------------------
# import
#-----------------------------------------
import os
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation, MaxPooling2D, Dropout, UpSampling2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils.data_utils import get_file

#-----------------------------------------
# defines
#-----------------------------------------
WEIGHTS_VGG16_FNAME = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_VGG16 = os.path.join(
    "https://github.com/fchollet/deep-learning-models/releases/download/v0.1", WEIGHTS_VGG16_FNAME)


#-----------------------------------------
# functions
#-----------------------------------------
def get_vgg16_weights_path():
    return get_file(WEIGHTS_VGG16_FNAME, WEIGHTS_VGG16, cache_subdir='models')


def build_vgg16(input_shape=(224, 224, 3), weights_path=None, weight_decay=0., drop_rate=None):

    model_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', kernel_regularizer=l2(weight_decay))(model_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', kernel_regularizer=l2(weight_decay))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Conver dense layer to conv2d layer
    x = Conv2D(4096, (7, 7), activation='relu', padding='same',
               name='fc6', kernel_regularizer=l2(weight_decay))(x)
    if drop_rate:
        x = Dropout(drop_rate, name='drop_out_fc6')(x)

    x = Conv2D(4096, (1, 1), activation='relu', padding='same',
               name='fc7', kernel_regularizer=l2(weight_decay))(x)

    if drop_rate:
        x = Dropout(drop_rate, name='drop_out_fc7')(x)

    model = Model(model_input, x)

    vgg16_weights_path = weights_path or get_vgg16_weights_path()
    # print(vgg16_weights_path)

    model.load_weights(vgg16_weights_path, by_name=True)

    return model
