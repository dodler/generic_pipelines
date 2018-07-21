import os

import cv2
import os.path as osp
from torch.utils.data.dataset import Dataset


def get_files_with_paths(base, target):
    target_path = osp.join(base, target)
    files = sorted(os.listdir(target_path))
    return list(map(lambda e: osp.join(target_path, e), files))


class IsicDataset(Dataset):

    def __init__(self, path, train_transform, val_transform):
        self.path = path
        self.train_imgs = get_files_with_paths(path, 'ISIC-2017_Training_Data')
        self.train_masks = get_files_with_paths(path, 'ISIC-2017_Training_Part1_GroundTruth')
        self.val_imgs = get_files_with_paths(path, 'ISIC-2017_Validation_Data')
        self.val_masks = get_files_with_paths(path, 'ISIC-2017_Validation_Part1_GroundTruth')

        print(len(self.train_imgs), len(self.train_masks), len(self.val_imgs), len(self.val_masks))

        self.mode = 'train'
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.cache = {}

    def get_key(self, index):
        return self.mode + str(index)

    def check_cache(self, index):
        k = self.get_key(index)
        if k in self.cache.keys():
            return self.cache[k]

        return None, None

    def __getitem__(self, index):
        if self.mode == 'train':
            tr = self.train_transform
            img = self.train_imgs[index]
            mask = self.train_masks[index]
        else:
            tr = self.val_transform
            img = self.val_imgs[index]
            mask = self.val_masks[index]

        img_c, mask_c = self.check_cache(index)
        if img_c is not None and mask_c is not None:
            return tr(img_c, mask_c)
        else:
            img = cv2.imread(img)
            mask = self.binarize(cv2.imread(mask, cv2.IMREAD_GRAYSCALE))
            self.put_to_cache(index, img, mask)
            return tr(img, mask)

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
            # return 16
        return len(self.val_imgs)
        # return 16

    def set_mode(self, mode):
        self.mode = mode

    def put_to_cache(self, index, img, mask):
        k = self.get_key(index)
        self.cache[k] = img, mask

    def binarize(self, mask):
        m = mask.copy()
        m[m > 0] = 1
        return m
