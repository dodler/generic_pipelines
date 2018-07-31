import random


class FirstImage(object):
    def __call__(self, img_batch):
        return img_batch.detach().squeeze(0).cpu().numpy()[0]

class ImageByIndex(object):
    def __call__(self, img_batch, index):
        return img_batch.detach().squeeze(0).cpu().numpy()[index]

class RandomImage(object):
    def __call__(self, img_batch):
        return random.choice(img_batch.detach().squeeze(0).cpu().numpy())
