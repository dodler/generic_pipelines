import os

import os.path as osp
import torch
from sklearn.model_selection import train_test_split
from abc import abstractmethod
from reader.image_reader import PillowReader


# todo add testing
class GenericXYDataset(object):

    def __init__(self, x_train_transform, x_val_transform,
                 y_train_transform, y_val_transform,
                 xy_paths_provider,
                 x_reader=PillowReader(), y_reader=PillowReader()):
        super().__init__()

        if x_train_transform is None or x_val_transform is None:
            raise AttributeError('Broken transform')

        self.mode = 'train'
        self.reader = x_reader
        X = []
        y = []

        for x_path, y_path in xy_paths_provider:
            X.append(x_path)
            y.append(y_path)

        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(X, y)

        self.x_train_transform = x_train_transform
        self.x_val_transform = x_val_transform
        self.y_train_transform = y_train_transform
        self.y_val_transform = y_val_transform
        self.x_cache = {}
        self.y_cache = {}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_y)
        else:
            return len(self.val_y)

    def setmode(self, mode):
        self.mode = mode

    @abstractmethod
    def read_x(self, index):
        pass

    @abstractmethod
    def read_y(self, index):
        pass

    def select_transform(self):
        if self.mode == 'train':
            return self.x_train_transform, self.y_train_transform
        else:
            return self.x_val_transform, self.y_val_transform

    def fetch_cache(self, index):
        key = self.mode + str(index)
        return self.x_cache[key], self.y_cache[key]

    def put_cache(self, index, X, y):
        key = self.mode + str(index)
        self.x_cache[key] = X
        self.y_cache[key] = y

    def __getitem__(self, index):
        if self.in_cache(index):
            X,y = self.fetch_cache(index)
        else:
            X = self.read_x(index)
            y = self.read_y(index)
            self.put_cache(index, X,y)

        x_transform, y_transform = self.select_transform()

        return x_transform(X), y_transform(y)

