#-----------------------------------------
# import
#-----------------------------------------
import os
import argparse
from keras.applications import VGG16
from keras.layers import Input, Conv2D, Conv2DTranspose, Add, Activation, MaxPooling2D
from keras.models import Model
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


# def FCN8s(classes=21, input_shape=(224, 224, 3)):

#     vgg16 = VGG16(weights='imagenet', include_top=False,
#                 input_shape=input_shape)

#     input = Input(shape=input_shape)

#     '''
#     Base Encorder VGG16
#     '''
#     for layer in vgg16.layers[1:19]:
#         if layer.name == 'block1_conv1':
#             x = layer(input)
#         else:
#             x = layer(x)

#         if layer.name == 'block3_pool':
#             p3 = x
#         elif layer.name == 'block4_pool':
#             p4 = x
#         elif layer.name == 'block5_pool':
#             p5 = x
#         else:
#             pass

#     '''
#     Skip Connection
#     '''
#     # dimention reduction
#     p3 = Conv2D(classes, 1, activation='relu', name='conv_p3')(p3)
#     p4 = Conv2D(classes, 1, activation='relu', name='conv_p4')(p4)
#     p5 = Conv2D(classes, 1, activation='relu', name='conv_p5')(p5)

#     # upsampling x2
#     u4 = Conv2DTranspose(classes, 4, activation='relu',
#                         strides=2, padding='same', name='upscore_p4')(p4)
#     # upsampling x4
#     u5 = Conv2DTranspose(classes, 8, activation='relu',
#                         strides=4, padding='same', name='upscore_p5')(p5)

#     x = Add(name='add')([p3, u4, u5])

#     # upsampling x8
#     x = Conv2DTranspose(classes, 16, activation='relu',
#                         strides=8, padding='same', name='upscore_final')(x)

#     x = Activation('softmax', name='softmax')(x)

#     model = Model(input, x)

#     return model


def FCN8s(classes=21, input_shape=(224, 224, 3)):

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

    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(x)
    #x = Dropout(0.5)(x)
    p5 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    #x = Dropout(0.5)(x)

    '''
    Skip Connection
    '''
    # dimention reduction
    p3 = Conv2D(classes, 1, activation='relu', name='conv_p3')(p3)
    p4 = Conv2D(classes, 1, activation='relu', name='conv_p4')(p4)
    p5 = Conv2D(classes, 1, activation='relu', name='conv_p5')(p5)

    # upsampling x2
    u4 = Conv2DTranspose(classes, 4, activation='relu',
                         strides=2, padding='same', name='upscore_p4')(p4)
    # upsampling x4
    u5 = Conv2DTranspose(classes, 8, activation='relu',
                         strides=4, padding='same', name='upscore_p5')(p5)

    x = Add(name='add')([p3, u4, u5])

    # upsampling x8
    x = Conv2DTranspose(classes, 16, activation='relu',
                        strides=8, padding='same', name='upscore_final')(x)

    x = Activation('softmax', name='softmax')(x)

    return Model(input, x)

#-----------------------------------------
# main
#-----------------------------------------


def main(args):
    model = FCN8s()
    print(model.summary())


if __name__ == '__main__':
    print("start")
    main(get_args())
    print("end")
