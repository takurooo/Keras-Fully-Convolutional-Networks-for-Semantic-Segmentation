# -------------------------------------------
# import
# -------------------------------------------
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------
# defines
# -------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

# -------------------------------------------
# private functions
# -------------------------------------------


def is_landscape(img_width, img_height):
    return img_width >= img_height


def scale(img_pil, target_size):
    img_w, img_h = img_pil.size
    target_w, target_h = target_size
    if is_landscape(img_w, img_h):
        ratio = img_w//img_h
        img_pil = img_pil.resize((target_w*ratio, target_h))
    else:
        ratio = img_h//img_w
        img_pil = img_pil.resize((target_w, target_h*ratio))
    return img_pil

# -------------------------------------------
# public functions
# -------------------------------------------


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, seed=None):
        for t in self.transforms:
            if seed:
                random.seed(seed)
            x = t(x)
        return x


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_pil, seed=None):
        if seed:
            random.seed(seed)

        flip = random.random()
        if flip < self.p:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        return img_pil


class RandomCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def get_random_box(self, img_size, output_size):
        img_w, img_h = img_size
        out_w, out_h = output_size
        if img_w == out_w and img_h == out_h:
            return 0, 0, img_w, img_h

        left = random.randint(0, img_w - out_w)
        upper = random.randint(0, img_h - out_h)
        right = left + out_w
        lower = upper + out_h

        return (left, upper, right, lower)

    def __call__(self, img_pil, seed=None):
        img_w, img_h = img_pil.size

        if min(img_w, img_h) < min(self.crop_size):
            img_pil = scale(img_pil, self.crop_size)

        if seed:
            random.seed(seed)
        crop_box = self.get_random_box(img_pil.size, self.crop_size)
        img_pil = img_pil.crop(crop_box)
        return img_pil


class CenterCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def get_center_box(self, img_size, output_size):
        img_w, img_h = img_size
        out_w, out_h = output_size

        left = (img_w - out_w) // 2
        upper = (img_h - out_h) // 2
        right = left + out_w
        lower = upper + out_h

        return (left, upper, right, lower)

    def __call__(self, img_pil):
        img_w, img_h = img_pil.size

        if min(img_w, img_h) < min(self.crop_size):
            img_pil = scale(img_pil, self.crop_size)

        crop_box = self.get_center_box(img_pil.size, self.crop_size)
        return img_pil.crop(crop_box)


# -------------------------------------------
# main
# -------------------------------------------


if __name__ == '__main__':
    pass
