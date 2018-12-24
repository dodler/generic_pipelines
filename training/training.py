import logging as l
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import *

from tensorboardX import SummaryWriter
from generic_utils.utils import AverageMeter


def create_logger(file_name):
    logFormatter = l.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = l.getLogger()
    logger.setLevel(l.DEBUG)

    fileHandler = l.FileHandler(file_name)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    return logger


class Trainer(object):

    def __init__(self, criterion,
                 metric,
                 optimizer,
                 model_name,
                 model,
                 base_checkpoint_name=None,
                 device=0,
                 dummy_input=None):
        '''

        :param watcher_env: environment for visdom
        :param criterion - loss function
        '''
        if base_checkpoint_name is None:
            self.base_checkpoint_name = model_name
        else:
            self.base_checkpoint_name = base_checkpoint_name

        self.metric = metric
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True)
        self.best_loss = np.inf
        self.model_name = model_name
        self.device = device
        self.epoch_num = 0
        self.model = model

        self.logger = create_logger(model_name + '.log')
        self.writer = SummaryWriter(log_dir='/tmp/runs/')
        self.counters = {}

        if dummy_input is not None:
            self._plot_graph(dummy_input)

    @staticmethod
    def save_checkpoint(state, name):
        print('saving state at', name)
        torch.save(state, name)

    def get_checkpoint_name(self, loss):
        return self.base_checkpoint_name + '_best.pth.tar'

    def is_best(self, avg_loss):
        best = avg_loss < self.best_loss
        if best:
            self.best_loss = avg_loss

        return best

    def validate(self, val_loader):
        batch_time = AverageMeter()
        losses = AverageMeter()
        metrics = AverageMeter()

        self.model.eval()

        end = time.time()
        tqdm_val_loader = tqdm(enumerate(val_loader))
        for batch_idx, (input, target) in tqdm_val_loader:
            with torch.no_grad():
                input_var = input.to(self.device)
                target_var = target.to(self.device)

                output = self.model(input_var)

                loss = self.criterion(output, target_var)
                loss_scalar = loss.item()
                losses.update(loss_scalar)
                metric_val = self.metric(output, target_var)
                metrics.update(metric_val)
                tqdm_val_loader.set_description('val loss:%s, val metric: %s' %
                                                (str(loss_scalar), str(metric_val)))

            batch_time.update(time.time() - end)

            self._log_data(input, target, output, 'val_it_data')
            self._log_metric({
                'metric': metric_val,
                'loss': loss_scalar,
                'batch_time': time.time() - end
            }, 'val_it_metric')

            end = time.time()

        self._log_metric({
            'metric': metrics.avg,
            'loss': losses.avg,
            'batch_time': batch_time.avg
        }, 'val_epoch_metric')

        self.scheduler.step(losses.avg)

        if self.is_best(losses.avg):
            self.save_checkpoint(self.model.state_dict(), self.get_checkpoint_name(losses.avg))

        self.epoch_num += 1
        return losses.avg, metrics.avg

    def update_train_epoch_stats(self, loss, metric):
        self.epoch_train_losses.append(loss)
        self.epoch_train_metrics.append(metric)

    def train(self, train_loader):
        batch_time, data_time, losses, metric = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        self.model.train()

        end = time.time()
        train_tqdm_iterator = tqdm(enumerate(train_loader))
        for batch_idx, (input, target) in train_tqdm_iterator:
            data_time.update(time.time() - end)

            input_var = input.to(self.device)
            target_var = target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            loss.backward()
            self.optimizer.step()

            loss_scalar = loss.item()
            losses.update(loss_scalar)
            metric_val = self.metric(output, target_var)  # todo - add output dimention assertion
            metric.update(metric_val)
            train_tqdm_iterator.set_description('train loss:%s, train metric: %s' %
                                                (str(loss_scalar), str(metric_val)))

            batch_time.update(time.time() - end)
            end = time.time()

            self._log_data(input, target, output, 'train_it_data')
            self._log_metric({
                'metric': metric_val,
                'loss': loss_scalar,
                'batch_time': time.time() - end
            }, 'train_it_metric')

        self._log_metric({
            'metric': metric.avg,
            'loss': losses.avg,
            'batch_time': batch_time.avg
        }, 'train_epoch_metric')
        return losses.avg, metric.avg

    def _log_data(self, input, target, output, tag):
        it = self._get_it(tag)
        if it % 100 == 0:
            self.writer.add_image(tag, input[:, 0:3, :, :], it)

    def _log_metric(self, metrics_dict, tag):
        it = self._get_it(tag)

        result = 'tag: ' + tag
        for k in metrics_dict:
            self.writer.add_scalar(tag + '_' + k, metrics_dict[k], it)
            result += ' ,' + k + '=' + str(metrics_dict[k])

        result += ', iteration ' + str(it)

        self.logger.debug(result)

    def _get_it(self, tag):
        if tag in self.counters.keys():
            result = self.counters[tag]
            self.counters[tag] += 1
            return result
        else:
            self.counters[tag] = 0
            return 0

    def _plot_graph(self, dummy_input):
        self.writer.add_graph(self.model, dummy_input)
