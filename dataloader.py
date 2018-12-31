#-------------------------------------------
# import
#-------------------------------------------
import os
import re
import codecs
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils.list_util import *
#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

#-------------------------------------------
# private functions
#-------------------------------------------


def to_crop_center_box(img_width, img_height, crop_width, crop_height):
    xmin = (img_width - crop_width) // 2
    ymin = (img_height - crop_height) // 2
    xmax = xmin + crop_width
    ymax = ymin + crop_height

    return (xmin, ymin, xmax, ymax)


def is_landscape(img_width, img_height):
    return img_width >= img_height


def crop_square(img_pil, crop_size):
    img_width, img_height = img_pil.size

    if min(img_width, img_height) < crop_size:
        if is_landscape(img_width, img_height):
            ratio = img_width//img_height
            img_pil = img_pil.resize((crop_size*ratio, crop_size))
        else:
            ratio = img_height//img_width
            img_pil = img_pil.resize((crop_size, crop_size*ratio))
    img_width, img_height = img_pil.size
    crop_box = to_crop_center_box(img_width, img_height, crop_size, crop_size)

    return img_pil.crop(crop_box)


def one_hot_array(label_array, classes, size):
    x = np.zeros((size, size, classes))
    for i in range(size):
        for j in range(size):
            x[i, j, label_array[i][j]] = 1

    return x


def preprocess(x):
    # return x / 127.5 - 1
    return x / 255

#-------------------------------------------
# public functions
#-------------------------------------------


class DataLoader:

    def __init__(self, classes, input_size, preprocess_func=None):
        self.classes = classes
        self.input_size = input_size
        self.x_data_list = []
        self.y_data_list = []
        if preprocess_func:
            self.preprocess_func = preprocess_func
        else:
            self.preprocess_func = preprocess

    def load_data(self, path, label_data=False):
        img_pil = Image.open(path)
        img_pil = crop_square(img_pil, self.input_size)
        # img_pil.show()
        if label_data:
            img_array = np.asarray(img_pil, dtype=np.int32)
            img_array[img_array == 255] = 0  # 境界部分をbackgroundクラスにする
            img_array = one_hot_array(img_array, self.classes, self.input_size)
            img_array = np.expand_dims(img_array, axis=0)
        else:
            img_array = np.asarray(img_pil, dtype=np.float32)
            img_array = self.preprocess_func(img_array)  # for vgg16
            img_array = np.expand_dims(img_array, axis=0)
            # img_array = preprocess_input(img_array, mode='tf')  # for vgg16

        return img_array

    def flow_from_directory(self, x_dir, y_dir=None, batch_size=32, random_seed=None):
        x_paths = list_from_dir(x_dir, ('.jpg', '.png'))
        if y_dir is not None:
            y_paths = list_from_dir(y_dir, ('.jpg', '.png'))
        else:
            y_paths = None

        return self.flow(x_paths, y_paths, batch_size, random_seed)

    def flow(self, x_paths, y_paths=None, batch_size=32, random_seed=None):

        if y_paths is None:
            while True:
                x_paths_ = shuffle(x_paths,
                                   random_state=random_seed)
                for x_path in x_paths_:
                    # print(x_path)
                    x = self.load_data(x_path, label_data=False)
                    self.x_data_list.append(x[0])
                    if batch_size <= len(self.x_data_list):
                        x_data_list = np.asarray(
                            self.x_data_list, dtype=np.float32)
                        self.x_data_list = []
                        yield x_data_list
        else:
            while True:
                x_paths_, y_paths_ = shuffle(
                    x_paths, y_paths, random_state=random_seed)
                for x_path, y_path in zip(x_paths_, y_paths_):
                    # print(x_path)
                    # print(y_path)
                    x = self.load_data(x_path, label_data=False)
                    y = self.load_data(y_path, label_data=True)
                    self.x_data_list.append(x[0])
                    self.y_data_list.append(y[0])
                    if batch_size <= len(self.x_data_list):
                        x_data_list = np.asarray(
                            self.x_data_list, dtype=np.float32)
                        y_data_list = np.asarray(
                            self.y_data_list, dtype=np.float32)
                        self.x_data_list = []
                        self.y_data_list = []
                        yield x_data_list, y_data_list


#-------------------------------------------
# main
#-------------------------------------------

if __name__ == '__main__':
    print("start")

    train_img_dir = os.path.join(CUR_PATH, "data", "train", "img")
    train_gt_dir = os.path.join(CUR_PATH, "data", "train", "gt")
    #train_img_paths = list_from_dir(train_img_dir, ('.jpg', '.png'))
    #train_gt_paths = list_from_dir(train_gt_dir, ('.jpg', '.png'))

    loader = DataLoader(classes=21, input_size=224, preprocess_func=None)
    for train, target in loader.flow_from_directory(train_img_dir, train_gt_dir, batch_size=1, random_seed=0):
        print(train.shape)
        print(target.shape)
        imgs = train
        gts = target
        break

    # img = imgs[0]
    # img = img.astype(np.uint8)
    # plt.figure()
    # plt.imshow(img)

    # gt = gts[0, :, :, 0]  # background class
    # plt.figure()
    # plt.imshow(gt)

    # plt.show()

    print("end")
