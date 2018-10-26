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
        self.env_name = env_name
        self.nrow = nrow
        self._wins = {}
        self.display_amount = display_amount
        self._vis = visdom.Visdom(env=env_name)

    def _display(self, img_batch, caption):
        if caption in self._wins.keys():
            self._vis.images(img_batch[:self.display_amount],
                             nrow=self.nrow,
                             win = self._wins[caption],
                             opts=dict(caption=caption))
        else:
            self._wins[caption] = self._vis.images(img_batch[:self.display_amount],
                                                   nrow=self.nrow,
                                                   opts=dict(caption=caption))

    def __call__(self, input, target, output):
        self._display(input, 'input')
        self._display(target, 'target')
        self._display(output, 'output')

    def close_windows(self):
        for k in self._wins.keys():
            self._vis.close(self._wins[k])
