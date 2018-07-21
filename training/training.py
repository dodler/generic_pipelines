import shutil
import time

import torch
from torch.autograd import Variable

from generic_utils.output_watchers import ClassificationWatcher
from generic_utils.segmentation import AverageMeter
from generic_utils.segmentation import VisdomValueWatcher

VAL_LOSS = 'val loss'
VAL_ACC = 'val acc'
TRAIN_ACC_OUT = 'train acc'
TRAIN_LOSS_OUT = 'train loss'


class Trainer(object):

    def __init__(self, watcher_env, criterion, metric):
        '''

        :param watcher_env: environment for visdom
        :param criterion - loss function
        '''
        self.metric = metric
        self.criterion = criterion
        self.watcher = VisdomValueWatcher(watcher_env)
        self.output_watcher = ClassificationWatcher(self.watcher)

    def set_output_watcher(self, output_watcher):
        self.output_watcher = output_watcher

    def get_watcher(self):
        return self.watcher

    def save_checkpoint(self, state, is_best, epoch, loss, filename='checkpoint_{}_loss_{}.pth.tar'):
        _filename = filename.format(epoch, loss)
        print('Save model at epoch {epoch} in {file}'.format(epoch=epoch + 1, file=_filename))
        torch.save(state, _filename)
        if is_best:
            shutil.copyfile(_filename, 'model_best.pth.tar')

    def validate(self, val_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for batch_idx, (input, target) in enumerate(val_loader):
            input_var = Variable(input.cuda(), volatile=True)
            target_var = Variable(target.cuda(), volatile=True)

            output = model(input_var)

            loss = self.criterion(output.view(-1), target_var.view(-1))

            # measure accuracy and record loss
            losses.update(loss.data[0], input.size(0))
            self.output_watcher(output)

            metric_val = self.metric(output, target_var.view(-1))
            acc.update(metric_val)

            self.watcher.log_value(VAL_ACC, metric_val)
            self.watcher.log_value(VAL_LOSS, loss.data[0])

            self.watcher.display_every_iter(batch_idx, input_var, target, output, 'validation')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('\rValidation: [{0}/{1}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'accuracy {acc.avg:.4f}\t'.format(
                batch_idx, len(val_loader), eta=batch_time.avg * len(val_loader),
                time=batch_time.sum, loss=losses, acc=acc), end='')
        print()
        return losses.avg, acc.avg

    def train(self, train_loader, model, optimizer, epoch):
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

            optimizer.zero_grad()
            output = model(input_var)
            loss = self.criterion(output.view(-1), target_var.view(-1))

            loss.backward()
            optimizer.step()

            self.watcher.display_labels_hist(target_var)
            self.watcher.display_most_confident_error(batch_idx, input_var, target_var, output)

            # measure accuracy and record loss

            losses.update(loss.data[0], input.size(0))

            self.output_watcher(output)

            metric_val = self.metric(output, target_var.view(-1)) # todo - add output dimention assertion
            acc.update(metric_val, batch_idx)

            self.watcher.log_value(TRAIN_ACC_OUT, metric_val)
            self.watcher.log_value(TRAIN_LOSS_OUT, loss.data[0])

            self.watcher.display_every_iter(batch_idx, input_var, target, output, 'train')

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('\rEpoch: {0}  [{1}/{2}]\t'
                  'ETA: {time:.0f}/{eta:.0f} s\t'
                  'data loading: {data_time.val:.3f} s\t'
                  'loss {loss.avg:.4f}\t'
                  'accuracy {acc.avg:.4f}\t'.format(
                epoch, batch_idx, len(train_loader), eta=batch_time.avg * len(train_loader),
                time=batch_time.sum, data_time=data_time, loss=losses, acc=acc), end='')
        return losses.avg, acc.avg
