#-------------------------------------------
# import
#-------------------------------------------
import os
import argparse
import importlib
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataloader import DataLoader
from utils import list_util
from utils.color import make_cmap

#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))
JSON_PATH = os.path.join(CUR_PATH, 'args.json')

N_CLASS = 21
INPUT_SIZE = 224
#-------------------------------------------
# private functions
#-------------------------------------------

'''
VOC2012 Classes
'''
# "background": 0,
# "aeroplane": 1,
# "bicycle": 2,
# "bird": 3,
# "boad": 4,
# "bottle": 5,
# "bus": 6,
# "car": 7,
# "cat": 8,
# "chair": 9,
# "cow": 10,
# "dining table": 11,
# "dog": 12,
# "horse": 13,
# "motor_bike": 14,
# "person": 15,
# "potted_plant": 16,
# "sheep": 17,
# "sofa": 18,
# "train": 19,
# "tv": 20


def get_args():
    with open(JSON_PATH, "r") as f:
        j = json.load(f)
    return j['predict']


def pred_to_img(pred):
    cmap = make_cmap()

    pred_ = np.argmax(pred[0], axis=2)
    row, col = pred_.shape
    dst = np.ones((row, col, 3))
    for i in range(21):
        dst[pred_ == i] = cmap[i]

    return np.uint8(dst)


def latest_weight(log_dir):
    weight_paths = list_util.list_from_dir(log_dir, '.hdf5')
    if len(weight_paths) == 0:
        return ""
    else:
        return weight_paths[-1]


def main(args):
    model_name = args["model"]
    img_dir = args["img_dir"]
    log_dir = args["log_dir"]
    img_idx = 33

    img_paths = list_util.list_from_dir(img_dir, ('.jpg', '.png'))

    print("img_len   : {}".format(len(img_paths)))

    '''
    Create DataLoader
    '''
    loader = DataLoader(N_CLASS, input_size=INPUT_SIZE)
    input_img = loader.load_data(img_paths[img_idx])

    '''
    Create Model
    '''
    model = importlib.import_module("models." + model_name)
    model = model.build(classes=N_CLASS,
                        input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # mode.summary()

    '''
    Load Weights
    '''
    weight_path = latest_weight(log_dir)
    if os.path.exists(weight_path):
        print('load weight : ', weight_path)
        model.load_weights(weight_path)

    '''
    Predict
    '''
    pred = model.predict(input_img)

    '''
    Convert
    '''
    pred_img = pred_to_img(pred)

    input_img_ = (input_img[0] + 1) * 127.5
    input_img_ = np.uint8(input_img_)

    '''
    Show Predict to Image
    '''
    plt.figure(figsize=(15, 15))
    img_list = [input_img_, pred_img]
    titel_list = ["input img", "predicted img"]
    plot_num = 1
    for title, img in zip(titel_list, img_list):
        plt.subplot(1, 2, plot_num)
        plt.title(title)
        plt.axis("off")
        plt.imshow(img)
        plot_num += 1

    # plt.show()
    plt.savefig("predict_{}.png".format(img_idx))


if __name__ == '__main__':
    main(get_args())
