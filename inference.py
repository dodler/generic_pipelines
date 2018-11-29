import cv2
import numpy as np


class TTA:
    def __init__(self, predict_func):
        self.predict_func = predict_func

    def __call__(self, x):
        hor_flip = cv2.flip(x, 1, x);

        rot90 = cv2.transpose(x, x);
        rot90 = cv2.flip(rot90, 1, rot90);

        rot270 = cv2.transpose(x,x);
        rot270 = cv2.flip(rot270, 0, rot270);

        rot180 = cv2.flip(x, -1, x);

        return self.predict_func(np.stack([hor_flip, rot90, rot180, rot270]))


