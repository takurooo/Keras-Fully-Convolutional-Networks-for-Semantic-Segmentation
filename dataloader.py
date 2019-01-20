# -------------------------------------------
# import
# -------------------------------------------
import os
import re
import codecs
import random
from PIL import Image
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from list_util import *
import transforms
# -------------------------------------------
# defines
# -------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

# -------------------------------------------
# private functions
# -------------------------------------------


# -------------------------------------------
# public functions
# -------------------------------------------


class Dataset:

    def __init__(self, classes, input_size, img_dir, label_dir=None, train=True):
        self.classes = classes
        self.input_size = input_size  # WH
        self.img_paths = list_from_dir(img_dir, ('.jpg', '.png'))
        self.train = train
        if label_dir:
            self.label_paths = list_from_dir(label_dir, ('.jpg', '.png'))
        else:
            self.label_paths = None

        if self.train:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(self.input_size)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.CenterCrop(self.input_size)
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_pil = Image.open(self.img_paths[idx])

        seed = random.randint(0, 2**32)
        img_pil = self.transforms(img_pil, seed=seed)
        img = self.format_img(img_pil)

        if self.label_paths:
            label_pil = Image.open(self.label_paths[idx])
            label_pil = self.transforms(label_pil, seed=seed)
            label = self.to_label(label_pil)
        else:
            label = None

        return img, label

    def format_img(self, img_pil):
        img_array = np.asarray(img_pil, dtype=np.float32)
        img_array = img_array / 255  # for vgg16
        img_array = np.expand_dims(img_array, axis=0)
        # img_array = preprocess_input(img_array, mode='tf')  # for vgg16
        return img_array

    def to_one_hot(self, label_array):
        w, h = self.input_size
        x = np.zeros((h, w, self.classes))
        for i in range(h):
            for j in range(w):
                x[i, j, label_array[i][j]] = 1
        return x

    def to_label(self, label_pil):
        label_array = np.asarray(label_pil, dtype=np.int32)
        label_array[label_array == 255] = 0  # 境界部分をbackgroundクラスにする
        label_array = self.to_one_hot(label_array)
        label_array = np.expand_dims(label_array, axis=0)
        return label_array


class DataLoader:

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_data_list = []
        self.y_data_list = []

    def __len__(self):
        return len(self.dataset)

    def flow(self):
        while True:
            data_num = len(self.dataset)
            if self.shuffle:
                indices = np.random.randint(0, data_num, data_num)
            else:
                indices = np.arange(data_num)
            for i in indices:
                img, label = self.dataset[i]
                self.x_data_list.append(img[0])
                self.y_data_list.append(label[0])
                if self.batch_size <= len(self.x_data_list):
                    x_data_list = np.asarray(
                        self.x_data_list, dtype=np.float32)
                    y_data_list = np.asarray(
                        self.y_data_list, dtype=np.uint8)
                    self.x_data_list = []
                    self.y_data_list = []
                    yield x_data_list, y_data_list


# -------------------------------------------
# main
# -------------------------------------------

if __name__ == '__main__':
    print("start")

    train_img_dir = os.path.join(CUR_PATH, "data", "train", "img")
    train_gt_dir = os.path.join(CUR_PATH, "data", "train", "gt")
    #train_img_paths = list_from_dir(train_img_dir, ('.jpg', '.png'))
    #train_gt_paths = list_from_dir(train_gt_dir, ('.jpg', '.png'))

    dataset = Dataset(classes=21, input_size=(224, 224),
                      img_dir=train_img_dir, label_dir=train_gt_dir,
                      train=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for train, target in loader.flow():
        print(train.shape, train[0].dtype)
        print(target.shape, target[0].dtype)
        imgs = train
        labels = target
        break

    img = imgs[0] * 255
    img = img.astype(np.uint8)
    plt.figure()
    plt.imshow(img)

    label = labels[0, :, :, 0]  # background class
    plt.figure()
    plt.imshow(label)

    plt.show()

    print("end")
