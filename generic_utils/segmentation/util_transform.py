import cv2
import numpy as np
import torch


class DualToTensor(object):
    def __call__(self, img, mask):
        img_tensor = torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1)
        return img_tensor, torch.ByteTensor(mask.astype(np.int))

class DualImgsToTensor(object):
    def __call__(self, img1, img2):
        img_tensor_1 = torch.from_numpy(img1[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1)
        img_tensor_2 = torch.from_numpy(img2[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1)
        return img_tensor_1,img_tensor_2


class DualResize(object):
    def __init__(self, target_shape):
        self._target_shape = target_shape

    def __call__(self, img, mask):
        return cv2.resize(img, self._target_shape), \
               cv2.resize(mask, self._target_shape)
