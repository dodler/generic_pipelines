class FirstImage(object):
    def __call__(self, img_batch):
        return img_batch.detach().squeeze(0).cpu().numpy()[0]

