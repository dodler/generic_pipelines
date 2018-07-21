import Image
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
