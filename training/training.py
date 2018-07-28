import shutil
import time
import numpy as np

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from generic_utils.output_watchers import ClassificationWatcher
from generic_utils.utils import AverageMeter
from generic_utils.visualization.visualization import VisdomValueWatcher

VAL_LOSS = 'val loss'
VAL_ACC = 'val metric'
TRAIN_ACC_OUT = 'train metric'
TRAIN_LOSS_OUT = 'train loss'


class Trainer(object):

    def __init__(self, watcher_env, criterion, metric, optimizer):
        '''

        :param watcher_env: environment for visdom
        :param criterion - loss function
        '''
        self.metric = metric
        self.criterion = criterion
        self.watcher = VisdomValueWatcher(watcher_env)
        self.output_watcher = ClassificationWatcher(self.watcher)
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(optimizer, 'min')
        self.best_loss = np.inf

    def set_output_watcher(self, output_watcher):
        self.output_watcher = output_watcher

    def get_watcher(self):
        return self.watcher

    def save_checkpoint(self, state, name):
        print('saving state at', name)
        torch.save(state, name)

    def get_checkpoint_name(self, loss):
        return self.model_name + '_loss_' + str(loss) + '.pth.tar'

    def is_best(self, avg_loss):
        best = avg_loss < self.best_loss
        if best:
            self.best_loss = avg_loss

        return best

    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            with torch.no_grad():
                input_var = Variable(input.cuda())
                target_var = Variable(target.cuda())

            output = model(input_var)

            loss = self.criterion(output.view(-1), target_var.view(-1))

            # measure accuracy and record loss
            losses.update(loss.detach(), input.size(0))

            metric_val = self.metric(output, target_var)
            acc.update(metric_val)

            self.watcher.log_value(VAL_ACC, metric_val)
            self.watcher.log_value(VAL_LOSS, loss.detach())
            self.watcher.display_every_iter(batch_idx, input_var, target, output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('\rValidation: [{0}/{1}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'metric {acc.avg:.4f}\t'.format(
                batch_idx, len(val_loader), eta=batch_time.avg * len(val_loader),
                time=batch_time.sum, loss=losses, acc=acc), end='')
        print()
        self.scheduler.step(losses.avg)

        if self.is_best(losses.avg):
            self.save_checkpoint(model.get_state_dict(), self.get_checkpoint_name(losses.avg))

        return losses.avg, acc.avg

    def train(self, train_loader, model, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for batch_idx, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            self.optimizer.zero_grad()
            output = model(input_var)
            loss = self.criterion(output.view(-1), target_var.view(-1))

            loss.backward()
            self.optimizer.step()

            losses.update(loss.detach(), input.size(0))

            metric_val = self.metric(output, target_var)  # todo - add output dimention assertion
            acc.update(metric_val, batch_idx)

            self.watcher.log_value(TRAIN_ACC_OUT, metric_val)
            self.watcher.log_value(TRAIN_LOSS_OUT, loss.detach())
            self.watcher.display_every_iter(batch_idx, input_var, target, output)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('\rEpoch: {0}  [{1}/{2}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'data loading: {data_time.val:.3f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'metric {acc.avg:.4f}\t'.format(
                epoch, batch_idx, len(train_loader), eta=batch_time.avg * len(train_loader),
                time=batch_time.sum, data_time=data_time, loss=losses, acc=acc), end='')
        return losses.avg, acc.avg
