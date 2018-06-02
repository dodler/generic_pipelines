import os

import os.path as osp
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


class FolderCachingDataset(object):
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_labels)
        else:
            return len(self.val_labels)

    def setmode(self, mode):
        self.mode = mode

    def __init__(self, path, train_transform, val_transform):
        super().__init__()

        self.mode = 'train'

        self.path = path
        self.classes = os.listdir(path)
        images = []
        labels = []

        for i, cls in enumerate(self.classes):
            t = os.listdir(osp.join(path, cls))
            t = list(map(lambda e: osp.join(path, cls, e), t))
            images.extend(t)
            labels.extend([i] * len(t))

        self.train, self.val, self.train_labels, self.val_labels = train_test_split(images, labels)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.cache = {}

    def get_num_classes(self):
        return len(self.classes)

    def get_img(self, index):
        k = str(index) + self.mode
        if k in self.cache.keys():
            return self.cache[k]
        else:
            if self.mode == 'train':
                img = Image.open(self.train[index]).convert('RGB')
            else:
                img = Image.open(self.val[index]).convert('RGB')

        result = img.copy()
        img.close()
        self.cache[k] = result
        return result

    def __getitem__(self, index):

        img = self.get_img(index)

        if self.mode == 'train':
            label = self.train_labels[index]
            img = self.train_transform(img)
        else:
            label = self.val_labels[index]
            img = self.val_transform(img)

        return img, torch.LongTensor([label])


class HierarchicalDataset(object):
    def __init__(self, path, train_transform, test_transform):
        from sklearn.model_selection import train_test_split

        self.baseclasses = {}
        self.subclasses = {}
        self.files = []
        self.labels = []
        idx = 0
        for base_cl in os.listdir(path):
            for sc in os.listdir(osp.join(path, base_cl)):
                for img in os.listdir(osp.join(path, base_cl, sc)):
                    self.files.append(osp.join(path, base_cl, sc, img))
                    self.labels.append(idx)

                self.baseclasses[idx] = base_cl
                self.subclasses[idx] = sc
                idx += 1

        print(self.baseclasses, self.subclasses)

        self.train, self.test, self.train_l, self.test_l = train_test_split(self.files, self.labels)
        self.mode = 'train'

        self.cache = {}

        self.train_transform = train_transform
        self.test_transform = test_transform

    def setmode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_l)
        else:
            return len(self.test_l)

    def get_img(self, index):
        k = str(index) + self.mode
        if k in self.cache.keys():
            return self.cache[k]
        else:
            if self.mode == 'train':
                img = Image.open(self.train[index]).convert('RGB')
            else:
                img = Image.open(self.test[index]).convert('RGB')

        result = img.copy()
        img.close()
        self.cache[k] = result
        return result

    def __getitem__(self, index):

        img = self.get_img(index)

        if self.mode == 'train':
            label = self.train_l[index]
            img = self.train_transform(img)
        else:
            label = self.test_l[index]
            img = self.test_transform(img)

        return img, torch.LongTensor([label])

    def base_classes(self):
        return self.baseclasses

    def sub_classes(self):
        return self.subclasses
