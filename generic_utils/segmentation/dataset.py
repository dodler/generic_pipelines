from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import os.path as osp
import time
from PIL import Image
import cv2
import numpy as np
from tqdm import *
import json


class JsonNamesProvider(object):
    def __init__(self, path2json):
        self._path2json = path2json
        self._images = []
        self._masks = []
        with open(path2json, 'r') as jsf:
            raw_json = json.load(jsf)
            self.paths = raw_json.keys()

    def provide(self):
        return list(self.paths)


class JsonSegmentationDataset(object):
    def __init__(self, base, path2json, transform):
        self._transform = transform
        self._path2json = path2json
        self._base = base
        self._images = []
        self._masks = []
        with open(path2json, 'r') as jsf:
            self._raw_json = json.load(jsf)
        self.load()

    def load(self):
        img_names = list(self._raw_json.keys())
        missing_rate = 0
        for i in tqdm(range(len(img_names))):
            img_name = img_names[i]
            mask_name = self._raw_json[img_name]
            img_path = self._base + str(img_name)

            if not os.path.exists(img_path):
                #                print(img_path)
                missing_rate += 1
                continue
            img = cv2.imread(img_path).astype(np.float32)

            self._images.append(img.copy())
            mask = binarize(cv2.imread(self._base + mask_name,
                                       cv2.IMREAD_GRAYSCALE).astype(np.float32))
            if not np.all(np.equal(np.unique(mask), np.array([0, 1], dtype=np.float32))):
                print(mask_name)

            self._masks.append(mask.copy())

        print('missing rate:', missing_rate / float(len(self)))

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._transform(self._images[index], self._masks[index])


def binarize(mask):
    mask[mask > 0] = 1
    return mask


class InMemoryImgSegmDataset(Dataset):
    def __init__(self, path, img_path, mask_path,
                 train_transform, test_transform,
                 names_provider=None,
                 limit_len=-1):
        """
        :param path: path to directory with images and masks directories
        :param img_path: name of directory with images
        :param mask_path: name of directory with masks
        :param train_transform: dual transform applied to train part
        :param test_transform: dual transform applied to test part
        :param limit_len: how many images load in memory
        """
        self._train_images = []
        self._test_masks = []
        self._train_masks = []
        self._test_images = []
        self._test_transform = test_transform
        self._train_transform = train_transform
        self._path = path
        self._mode = 'train'

        self._names_provider = names_provider

        if names_provider is None:
            self._img_paths = os.listdir(osp.join(path, img_path))
        else:
            self._img_paths = names_provider.provide()

        self.train, self.test = train_test_split(self._img_paths)
        self._limit_len = limit_len
        self._img_path = img_path
        self._mask_path = mask_path
        self.load()

    def set_mode(self, mode):
        self._mode = mode

    def __len__(self):
        if self._limit_len != -1:
            return self._limit_len
        elif self._mode == 'train':
            return len(self._train_images)
        else:
            return len(self._test_images)

    def __getitem__(self, index):
        return self.getitemfrom(index)

    def load(self):
        missing = []
        print('start loading')
        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.train)

        for i in tqdm(range(target_len)):
            base_name = self.train[i].split('.')[0]  # name without extension
            im_p = osp.join(self._path, self._img_path, base_name + '.jpg')
            m_p = osp.join(self._path, self._mask_path, base_name + '_mask.tif')

            if not osp.exists(m_p) or not osp.exists(im_p):
                missing.append(im_p)
                continue

            img = cv2.imread(im_p).astype(np.float32)
            self._train_images.append(img.copy())

            mask = binarize(cv2.imread(m_p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            if not np.all(np.equal(np.unique(mask), np.array([0, 1], dtype=np.float32))):
                print(m_p)

            self._train_masks.append(mask)

        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.test)

        for i in tqdm(range(target_len)):
            base_name = self.test[i].split('.')[0]  # name without extension
            im_p = osp.join(self._path, self._img_path, base_name + '.jpg')
            m_p = osp.join(self._path, self._mask_path, base_name + '_mask.tif')

            if not osp.exists(m_p) or not osp.exists(im_p):
                missing.append(im_p)
                continue

            img = cv2.imread(im_p).astype(np.float32)
            self._test_images.append(img.copy())
            mask = binarize(cv2.imread(m_p, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            if not np.all(np.equal(np.unique(mask), np.array([0, 1], dtype=np.float32))):
                print('warning ', m_p, ' is not binary')
            self._test_masks.append(mask)

        print(missing)

    def getitemfrom(self, index):
        if self._mode == 'train':
            return self._train_transform(self._train_images[index], self._train_masks[index])

        return self._test_transform(self._test_images[index], self._test_masks[index])


class InMemoryImgDataset(Dataset):
    def __init__(self, path, limit_len=-1):
        self._path = path
        self._mode = 'train'
        self._img_paths = os.listdir(path)
        self._limit_len = limit_len
        self.__load__()

    def __len__(self):
        if self._limit_len != 1:
            return self._limit_len
        elif self._mode == 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, index):
        return self.__getitemfrom__(index, self._mode)

    def __load__(self):
        self.train, self.test = train_test_split(self._img_paths)
        self._train_images = []
        self._test_images = []

        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.train)

        for i in range(target_len):
            if i % 1000 == 0:
                print('loaded ', str(i))
                time.sleep(0.1)
            img = Image.open(osp.join(self._path, self.train[i]))
            self._train_images.append(img.copy())
            img.close()

        if self._limit_len != -1:
            target_len = self._limit_len
        else:
            target_len = len(self.test)

        for i in range(target_len):
            if i % 1000 == 0:
                print('loaded test', str(i))
                time.sleep(0.1)
            img = Image.open(osp.join(self._path, self.test[i]))
            self._test_images.append(img.copy())
            img.close()

    def __getitemfrom__(self, index, mode):
        if mode == 'train':
            return self._train_images[index],
        return self._test_images[index]
