import torch
import visdom


class ClassificationWatcher(object):
    def __init__(self, watcher):
        self.watcher = watcher

    def __call__(self, output):
        key = 'prediction_distribution'
        _, pred_classes = output.max(1)
        if pred_classes is not torch.Tensor:
            if len(pred_classes) <= 1:
                return
            self.watcher.display_hist_and_add(pred_classes.view(-1).cpu().data.numpy(), key,
                                              title='prediction labels distribution')
        else:
            self.watcher.display_hist_and_add(pred_classes, key, 'prediction labels distribution')


class RegressionWatcher(object):
    def __init__(self, watcher):
        self.watcher = watcher

    def __call__(self, output):
        key = 'regression_output'
        if output is not torch.Tensor:
            self.watcher.display_hist_and_add(output.view(-1).cpu().data.numpy(), key,
                                              title='regression prediction distribution')
        else:
            self.watcher.display_hist_and_add(output, key, 'regression prediction distribution')


class DisplayImage:
    def __init__(self, env_name, display_amount, nrow=2):
        self.nrow = nrow
        self.wins = {}
        self.display_amount = display_amount
        self._vis = visdom.Visdom(env=env_name)
        self._vis.delete_env(env_name)

    def _display(self, img_batch, caption):
        if caption in self.wins.keys():
            self._vis.images(img_batch[:self.display_amount],
                             nrow=self.nrow,
                             win = self.wins[caption],
                             opts=dict(caption=caption))
        else:
            self.wins[caption] = self._vis.images(img_batch[:self.display_amount],
                             nrow=self.nrow,
                             opts=dict(caption=caption))

    def __call__(self, input, output):
        self._display(input, 'input')
        self._display(output, 'output')
