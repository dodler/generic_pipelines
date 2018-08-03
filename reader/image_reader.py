from PIL import Image
import cv2


class PillowReader(object):
    def __call__(self, path):
        t = Image.open(path)
        img = t.copy()
        t.close()
        return img


class OpencvReader(object):
    def __call__(self, path):
        return cv2.imread(path)


class OpencvGrayReader():
    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

class OpencvRGBReader(object):
    def __call__(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
