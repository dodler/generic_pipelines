from torch.optim import Adam


def make_adam(model, base_lr=1e-2, lr_num=3):
    '''
    creates adam optimizer with different learning rates from layer to layer
    the least are located to the begining the largest - to the end
    :param model:
    :param base_lr: start point lr
    :param lr_num: number of layers to devide lr
    :return:
    '''
    ch_size = len(list(model.children()))

    assert lr_num >= 2, "At least 2 groups are supported"
    assert lr_num < ch_size, "Number of learning rates should be less than number of children in model"

    if lr_num > 0:
        lrs_for_layers = []
        for i in range(ch_size):
            lrs_for_layers.append(base_lr)
            if (i + 1) % lr_num == 0:
                base_lr /= 10

        parameters_and_lrs = _create_parameters_and_lrs(i, lrs_for_layers, model)
        optimizer = Adam(parameters_and_lrs, base_lr)
        return optimizer
    else:
        return Adam(model.parameters(), base_lr)


def _create_parameters_and_lrs(lrs_for_layers, model):
    lrs = []
    for i, child in enumerate(model.children()):
        lrs.append({
            "params": child.parameters(),
            "lr": lrs_for_layers[-i]  # we re using
        })
    return lrs
