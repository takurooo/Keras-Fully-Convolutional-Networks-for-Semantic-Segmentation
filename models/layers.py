#-----------------------------------------
# import
#-----------------------------------------
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np


#-----------------------------------------
# defines
#-----------------------------------------

#-----------------------------------------
# functions
#-----------------------------------------
def get_kernel_size(factor):
    return 2 * factor - factor % 2


def bilinear_upsample_weights(scale_factor, number_of_classes):
    filter_size = get_kernel_size(scale_factor)
    factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]

    col = (1 - abs(og[0] - center) / factor)
    row = (1 - abs(og[1] - center) / factor)
    upsample_kernel = col * row

    # (w, h, in_ch, out_ch)
    weights = np.zeros((filter_size, filter_size, number_of_classes, number_of_classes),
                       dtype=np.float32)
    for i in range(number_of_classes):
        weights[:, :, i, i] = upsample_kernel
    return weights


class BilinearUpSampling2D(Layer):
    """Upsampling2D with bilinear interpolation."""

    def __init__(self, scale_factor=(2, 2), data_format=None, **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in ('channels_last', 'channels_first')
        self.data_format = data_format
        self.scale_factor = scale_factor  # (height_factor, width_factor)
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def _compute_output_size(self, input_shape):
        if self.data_format == 'channels_last':
            _, input_h, input_w, _ = input_shape
        else:
            _, _, input_h, input_w = input_shape
        input_size = (input_h, input_w)
        output_size = input_size * np.array(self.scale_factor).astype('int32')
        return output_size

    def compute_output_shape(self, input_shape):
        """Compute outputshape for next layer."""
        output_size = self._compute_output_size(input_shape)
        output_h, output_w = output_size
        if self.data_format == 'channels_last':
            return (input_shape[0], output_h, output_w, input_shape[3])
        else:
            return (input_shape[0], input_shape[1], output_h, output_w)

    def call(self, x):
        """Execute bilinear upsampling."""
        input_shape = K.int_shape(x)
        output_size = self._compute_output_size(input_shape)
        return tf.image.resize_bilinear(x, size=output_size)
