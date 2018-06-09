import os

import os.path as osp
import torch
from PIL import Image
from sklearn.model_selection import train_test_split


class PillowReader(object):
    def __call__(self, path):
        t = Image.open(path)
        img = t.copy()
        t.close()
        return img


class DefaultClassProvider(object):
    def __init__(self, path):
        self.objs = []
        self.lbls = []
        for i, cls in enumerate(os.listdir(path)):
            t = osp.join(path, cls, os.listdir(cls))
            t = [v for v in t if v.endswith('.jpg')]
            self.objs.extend(t)
            self.lbls.extend([i] * len(t))

    def __getitem__(self, item):
        return self.objs[item], self.lbls[item]

    def __len__(self):
        return len(self.lbls)


# todo add testing
class FolderCachingDataset(object):
    '''
    assumes that there is a following folder structure
    path
        class1
            img1
            ...
            imgn
        class2
            img1
            ...
            imgn
        ...
        classn
            img1
            ...
            imgn

    works in 2 modes:
    train, validation
    also performs automatic train-val split
    uses Pillow for image reading
    caches raw images
    '''

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_labels)
        else:
            return len(self.val_labels)

    def setmode(self, mode):
        self.mode = mode

    def __init__(self, train_transform, val_transform, path_class_provider,
                 reader=PillowReader()):
        super().__init__()

        if train_transform is None or val_transform is None:
            raise AttributeError('Broken transform')

        self.mode = 'train'
        self.reader = reader
        images = []
        labels = []

        for i, path, cls in enumerate(path_class_provider):
            images.append(path)
            labels.append(cls)

        self.train, self.val, self.train_labels, self.val_labels = train_test_split(images, labels)

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.cache = {}

    def get_img(self, index):
        k = str(index) + self.mode
        if k in self.cache.keys():
            return self.cache[k]
        else:
            if self.mode == 'train':
                img = self.reader(self.train[index]).convert('RGB')
            else:
                img = self.reader(self.val[index]).convert('RGB')

        self.cache[k] = img
        return img

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
