#-------------------------------------------
# import
#-------------------------------------------
import os
import sys
import argparse
import codecs
import importlib
import json
import numpy as np
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
from dataloader import DataLoader, Dataset
import list_util
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
    model_name = args["model"]
    train_img_dir = args["train_img_dir"]
    train_gt_dir = args["train_gt_dir"]
    val_img_dir = args["val_img_dir"]
    val_gt_dir = args["val_gt_dir"]
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    log_dir = args["log_dir"]

    os.makedirs(log_dir, exist_ok=True)

    '''
    Create DataLoader
    '''
    trn_dataset = Dataset(classes=21, input_size=(224, 224),
                          img_dir=train_img_dir, label_dir=train_gt_dir,
                          train=True)
    val_dataset = Dataset(classes=21, input_size=(224, 224),
                          img_dir=val_img_dir, label_dir=val_gt_dir,
                          train=False)
    train_loader = DataLoader(trn_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)

    steps_per_epoch = len(trn_dataset) // batch_size
    validation_steps = len(val_dataset) // batch_size

    print("train_img_len   : {}".format(len(trn_dataset)))
    print("val_img_len     : {}".format(len(val_dataset)))
    print("epochs          : ", epochs)
    print("batch_size      : ", batch_size)
    print("steps_per_epoch : ", steps_per_epoch)

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
    model = importlib.import_module("models." + model_name)
    model = model.build(classes=N_CLASS,
                        input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
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
        train_loader.flow(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_loader.flow(),
        validation_steps=validation_steps,
        callbacks=cbs,
        verbose=1)

    print("Saved weights ", log_dir)


if __name__ == '__main__':
    main(get_args())
