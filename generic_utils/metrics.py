from sklearn.metrics import mean_squared_error


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mse(output, target):
    y_true = target.data.cpu().squeeze().numpy()
    y_pred = output.data.cpu().squeeze().numpy()
    return mean_squared_error(y_true, y_pred)


def iou(pred, target, n_classes=1):
    pred_t = pred.view(-1).float()
    target = target.view(-1).float()

    inter = (pred_t * target).sum()
    union = (pred_t + target).sum()

    return (inter / union).cpu().data[0];



def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))
