import time

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from generic_utils.output_watchers import ClassificationWatcher
from generic_utils.utils import AverageMeter
from generic_utils.visualization.visualization import VisdomValueWatcher

import logging as l

logFormatter = l.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = l.getLogger()
logger.setLevel(l.DEBUG)

fileHandler = l.FileHandler('log.txt')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = l.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)


class Trainer(object):

    def __init__(self, criterion, metric, optimizer, model_name, base_checkpoint_name=None, device=0):
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
        self.watcher = VisdomValueWatcher(model_name)
        self.output_watcher = ClassificationWatcher(self.watcher)
        self.optimizer = optimizer
        self.scheduler = CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=1e6)
        self.best_loss = np.inf
        self.model_name = model_name
        self.device = device
        self.epoch_num = 0
        self.epoch_val_losses = []
        self.epoch_val_metrics = []
        self.epoch_train_losses = []
        self.epoch_train_metrics = []

        self.full_history = {}

    def set_output_watcher(self, output_watcher):
        self.output_watcher = output_watcher

    def get_watcher(self):
        return self.watcher

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

    def update_val_epoch_stat(self, loss, metric):
        self.epoch_val_losses.append(loss)
        self.epoch_val_metrics.append(metric)

    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        metrics = AverageMeter()

        model.eval()

        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                input_var = input.to(self.device)
                target_var = target.to(self.device)

                output = model(input_var)

                loss = self.criterion(output, target_var)
                losses.update(loss.item())
                metric_val = self.metric(output, target_var)
                metrics.update(metric_val)

                self.log_full_history(loss=loss,metric=metric_val)

                self.watcher.display_every_iter(batch_idx, input_var, target, output)

            batch_time.update(time.time() - end)
            end = time.time()

        self.log_epoch(batch_idx, batch_time, losses, metrics, val_loader)
        self.scheduler.step(losses.avg)

        if self.is_best(losses.avg):
            self.save_checkpoint(model.state_dict(), self.get_checkpoint_name(losses.avg))
            # pickle.dump(model, open(self.get_checkpoint_name(losses.avg), 'wb'))

        self.epoch_num += 1
        return losses.avg, metrics.avg

    def log_epoch(self, batch_idx, batch_time, losses, metrics, val_loader):
        logger.debug('\rValidation: [{0}/{1}]\t'
                     'ETA: {time:.0f}/{eta:.0f} s\t'
                     'loss {loss.avg:.4f}\t'
                     'metric {acc.avg:.4f}\t'.format(
            batch_idx, len(val_loader), eta=batch_time.avg * len(val_loader),
            time=batch_time.sum, loss=losses, acc=metrics), end='')
        self.update_val_epoch_stat(losses.avg, metrics.avg)
        self.watcher.log_metric_value(self.epoch_train_metrics[self.epoch_num],
                                      self.epoch_val_metrics[self.epoch_num], self.model_name)
        self.watcher.log_loss_value(self.epoch_train_losses[self.epoch_num],
                                    self.epoch_val_losses[self.epoch_num], self.model_name)

    def update_train_epoch_stats(self, loss, metric):
        self.epoch_train_losses.append(loss)
        self.epoch_train_metrics.append(metric)

    def train(self, train_loader, model, epoch):
        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        model.train()

        end = time.time()
        for batch_idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input_var = input.to(self.device)
            target_var = target.to(self.device)

            output = model(input_var)

            loss = self.criterion(output, target_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                losses.update(loss.item())
                metric_val = self.metric(output, target_var)  # todo - add output dimention assertion
                acc.update(metric_val)
                self.log_full_history(loss=loss,metric=metric_val)

            batch_time.update(time.time() - end)
            end = time.time()

        logger.debug('\rEpoch: {0}  [{1}/{2}]\t'
                     'ETA: {time:.0f}/{eta:.0f} s\t'
                     'data loading: {data_time.val:.3f} s\t'
                     'loss {loss.avg:.4f}\t'
                     'metric {acc.avg:.4f}\t'.format(
            epoch, batch_idx, len(train_loader), eta=batch_time.avg * len(train_loader),
            time=batch_time.sum, data_time=data_time, loss=losses, acc=acc))

        self.update_train_epoch_stats(losses.avg, acc.avg)
        return losses.avg, acc.avg

    def log_full_history(self, **kwargs):
        for k in kwargs.keys():
            if k in self.full_history.keys():
                self.full_history[k].append(kwargs[k])
            else:
                self.full_history[k] = [kwargs[k]]
