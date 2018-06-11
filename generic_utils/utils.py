import random

import cv2
import numpy as np
import torch
import visdom
from torch.autograd import Variable


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
    return torch.from_numpy(img[:, :, (2, 1, 0)].astype(np.float32)).permute(2, 0, 1)


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
        return cv2.cvtColor(tensor.data.numpy(), cv2.COLOR_RGB2BGR) # todo replace with permute


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
        s_w = random.randint(0, min(img.shape[0],self.crop_w))
        s_h = random.randint(0, min(img.shape[1],self.crop_h))
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


class VisdomValueWatcher(object):
    def __init__(self, env_name='main'):
        self._watchers = {}
        self._wins = {}
        self._vis = visdom.Visdom(env=env_name)
        self.vis_img = None
        self.vis_hist = None
        self.vis_conf_er = None

        self.wins = {}

    def display_and_add(self, img, title, key=None):
        if key is None:
            key = title
        if key in self.wins.keys():
            self._vis.image(img, opts=dict(title=title), win=self.wins[key])
        else:
            self.wins[key] = self._vis.image(img, opts=dict(title=title))

    def display_hist_and_add(self, vals, key, title=None):
        ''''
        vals - numpy array
        '''

        if key in self.wins.keys():
            self._vis.histogram(vals, win=self.wins[key], opts=dict(title=title))
        else:
            self.wins[key] = self._vis.histogram(vals, opts=dict(title=title))

    def display_labels_hist(self, labels):
        key = 'source_labels_histogram'
        numpy = labels.view(-1).cpu().data.numpy()
        if len(numpy) == 1:
            return
        self.display_hist_and_add(numpy, key, title='labels distribution')


    def display_most_confident_error(self, iter_num, img_batch, gt, batch_probs):
        if iter_num % 10 == 0:
            probs, cls = batch_probs.max(dim=1)
            gt = gt.data.cpu().numpy()
            probs = probs.data.cpu().numpy()
            cls = cls.data.cpu().numpy()

            bad_prob = -1
            bad_index = 0
            for i, prob in enumerate(probs):
                if cls[i] != gt[i] and prob > bad_prob:
                    bad_index = i
                    bad_prob = prob

            bad_cls_gt = gt[bad_index]
            bad_prob = probs[bad_index]
            bad_class = cls[bad_index]
            bad_img = img_batch.data.squeeze(0).cpu().numpy()[bad_index]

            if bad_prob > 0:
                self.display_and_add(bad_img,
                                     'error at gt:' + str(bad_cls_gt) + ', pred:' + str(bad_class) + ', prob:' + str(
                                         bad_prob),
                                     'confident_errors')

    def display_img_every(self, n, iter, image, key, title):
        '''
        display first image from variable or tensor in batch
        :param n:
        :param iter:
        :param image:
        :param key:
        :param title:
        :return:
        '''
        if iter % n == 0:
            if isinstance(image, Variable):
                img = image.data.squeeze(0).cpu().numpy()[0]
            else:
                img = image.squeeze(0).cpu().numpy()[0]
            self.display_and_add(img, title, key)

    def display_every_iter(self, iter_num, X, gt_class, pred_class, base_label):
        if iter_num % 10 == 0:
            img = X.data.squeeze(0).cpu().numpy()[0]
            _, pred = pred_class.max(dim=1)
            pred = pred.data.cpu().numpy()[0]
            gt = gt_class.cpu().numpy()[0]

            title = base_label + ' gt class ' + str(gt) + ', pred class ' + str(pred)
            self.display_and_add(img, title, 'display_preds')

    def get_vis(self):
        return self._vis

    def log_value(self, name, value, output=True):
        if name in self._watchers.keys():
            self._watchers[name].append(value)
        else:
            self._watchers[name] = [value]

        if output:
            self.output(name)

    def output_all(self):
        for name in self._wins.keys():
            self.output(name)

    def movingaverage(self, values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    def output(self, name):
        if name in self._wins.keys():

            y = self.movingaverage(self._watchers[name], 20)
            x = np.array(range(len(y)))

            self._vis.line(Y=y, X=x,
                           win=self._wins[name], update='new',
                           opts=dict(title=name))
        else:
            self._wins[name] = self._vis.line(Y=np.array(self._watchers[name]),
                                              X=np.array(range(len(self._watchers[name]))),
                                              opts=dict(title=name))

    def clean(self, name):
        self._watchers[name] = []
