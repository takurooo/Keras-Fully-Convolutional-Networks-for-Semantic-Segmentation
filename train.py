#-------------------------------------------
# import
#-------------------------------------------
import os
import argparse
import codecs
import json
import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from models import FCN8s
from dataloader import DataLoader
from utils import list_util

#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))
JSON_PATH = os.path.join(CUR_PATH, 'args.json')
WEIGHT_PATH = os.path.join(CUR_PATH, 'weights')
TRAIN_DATA_DIR = os.path.join(CUR_PATH, 'data', 'train')
VAL_DATA_DIR = os.path.join(CUR_PATH, 'data', 'val')
TRAIN_IMG_DIR = os.path.join(TRAIN_DATA_DIR, 'img')
TRAIN_GT_DIR = os.path.join(TRAIN_DATA_DIR, 'gt')
VAL_IMG_DIR = os.path.join(VAL_DATA_DIR, 'img')
VAL_GT_DIR = os.path.join(VAL_DATA_DIR, 'gt')

N_CLASS = 21
INPUT_SIZE = 224
#-------------------------------------------
# private functions
#-------------------------------------------


def get_args():
    with open(JSON_PATH, "r") as f:
        j = json.load(f)
    return j['train']


# def get_args():
#     parser = argparse.ArgumentParser(description='FCN via Keras')
#     parser.add_argument('img_dir',  type=str, help="img_data_dir")
#     parser.add_argument('gt_dir', type=str, help="target_data_dir")
#     return parser.parse_args()


def main(args):
    train_img_dir = args["train_img_dir"]
    train_gt_dir = args["train_gt_dir"]
    val_img_dir = args["val_img_dir"]
    val_gt_dir = args["val_gt_dir"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    log_dir = args["log_dir"]

    os.makedirs(log_dir, exist_ok=True)

    train_img_paths = list_util.list_from_dir(train_img_dir, ('.jpg', '.png'))
    train_gt_paths = list_util.list_from_dir(train_gt_dir, ('.jpg', '.png'))
    val_img_paths = list_util.list_from_dir(val_img_dir, ('.jpg', '.png'))
    val_gt_paths = list_util.list_from_dir(val_gt_dir, ('.jpg', '.png'))

    steps_per_epoch = len(train_img_paths) // batch_size
    validation_steps = len(val_img_paths) // batch_size

    print("train_img_len   : {}".format(len(train_img_paths)))
    print("train_gt_len    : {}".format(len(train_gt_paths)))
    print("val_img_paths   : {}".format(len(val_img_paths)))
    print("val_gt_paths    : {}".format(len(val_gt_paths)))
    print("epochs          : ", epochs)
    print("batch_size      : ", batch_size)
    print("steps_per_epoch : ", steps_per_epoch)

    '''
    Create DataLoader
    '''
    train_loader = DataLoader(N_CLASS, input_size=INPUT_SIZE)
    val_loader = DataLoader(N_CLASS, input_size=INPUT_SIZE)

    '''
    Setting Callback
    '''
    ckpt_name = 'weights-{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}-.hdf5'
    cbs = [
        ModelCheckpoint(os.path.join(log_dir, ckpt_name),
                        monitor='val_acc',
                        verbose=0,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='auto',
                        period=1)
    ]

    '''
    Create Model
    '''
    model = FCN8s(classes=N_CLASS, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
    # mode.summary()

    '''
    Compile
    '''
    optimizer = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    '''
    Start Training
    '''
    model.fit_generator(
        train_loader.flow(
            train_img_paths, train_gt_paths, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_loader.flow(
            val_img_paths, val_gt_paths, batch_size=batch_size),
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=1)

    print("Saved weights ", log_dir)


if __name__ == '__main__':
    main(get_args())
