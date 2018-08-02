import random

import cv2
import numpy as np
import torch


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val]
    return weight


def ocv2torch(img):
    '''
    bgr opencv image
    :param img:
    :return:
    '''
    return torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1) / 255.0


class OCVTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1)


class TensorToOCV(object):
    def __call__(self, tensor):
        '''
        no batch support yet
        :param tensor:
        :return:
        '''
        return cv2.cvtColor(tensor.data.numpy(), cv2.COLOR_RGB2BGR)  # todo replace with permute


class OCVResize(object):
    def __init__(self, w, h=None):
        if h is None:
            self.h = w

        self.w = w

    def __call__(self, img):
        print(img)
        return cv2.resize(img, (self.w, self.h))


class OCVRandomCrop(object):
    def __init__(self, crop_w, crop_h=None):
        self.crop_w = crop_w
        if crop_h is not None:
            self.crop_h = crop_h
        else:
            self.crop_h = crop_w

    def __call__(self, img):
        '''
        bgr ocv image
        :param img:
        :return:
        '''
        s_w = random.randint(0, min(img.shape[0], self.crop_w))
        s_h = random.randint(0, min(img.shape[1], self.crop_h))
        return img[s_w:s_w + self.crop_w, s_h:s_h + self.crop_h, :]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (1 + self.count)
