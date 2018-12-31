from keras.utils import plot_model
from models import *

model = vgg_fcn32s.build()
plot_model(model, to_file='vgg_fcn32s.png')

model = vgg_fcn16s.build()
plot_model(model, to_file='vgg_fcn16s.png')

model = vgg_fcn8s.build()
plot_model(model, to_file='vgg_fcn8s.png')
