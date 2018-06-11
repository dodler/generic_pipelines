import torch


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