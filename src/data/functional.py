from __future__ import division

import torch
import numbers
import collections
import numpy as np
from PIL import Image, ImageOps


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_mytensor(pic):
    pic_arr = np.array(pic)
    if pic_arr.ndim == 2:
        pic_arr = pic_arr[..., np.newaxis]
    img = torch.from_numpy(pic_arr.transpose((2, 0, 1)))
    if not isinstance(img, torch.FloatTensor):
        return img.float()  # no normalize .div(255)
    else:
        return img


def normalize(tensor, mean, std):
    if not _is_tensor_image(tensor):
        raise TypeError("tensor is not a torch image.")
    if tensor.size(0) == 1:
        tensor.sub_(mean).div_(std)
    else:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
    return tensor


def resize(img, size, interpolation=Image.BILINEAR):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))
    if not isinstance(size, int) and (not isinstance(size, collections.Iterable) or len(size) != 2):
        raise TypeError("Got inappropriate size arg: {}".format(size))

    if not isinstance(size, int):
        return img.resize(size[::-1], interpolation)

    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow = size
        oh = int(round(size * h / w))
    else:
        oh = size
        ow = int(round(size * w / h))
    return img.resize((ow, oh), interpolation)


def pad(img, padding, fill=0):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError("Got inappropriate fill arg")

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " + "{} element tuple".format(len(padding)))

    return ImageOps.expand(img, border=padding, fill=fill)


def crop(img, i, j, h, w):
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    return img.crop((j, i, j + w, i + h))
