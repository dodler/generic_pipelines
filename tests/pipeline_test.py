import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from training.training import Trainer

model = nn.Sequential(
    nn.Conv2d(kernel_size=2, out_channels=4, in_channels=3, stride=2),
    nn.AvgPool2d(2),
    nn.Conv2d(kernel_size=6, in_channels=4, out_channels=1, stride=2)
)

torch.manual_seed(42)
np.random.seed(42)


class TestDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn((3, 24, 24)), torch.zeros(1)


loss = nn.BCEWithLogitsLoss()


def mymetric(pred, target):
    preds = (pred > 0.1).int()
    targs = target.int()
    return (preds == targs).float().mean().item()


def myloss(pred, target):
    return loss(pred.reshape((2,1)), target)


LR = 1e-2
MODEL_NAME = 'test_model'
DEVICE = 'cpu'
BATCH_SIZE = 2
WORKERS = 2
EPOCHS = 20

train_ds = TestDataset(2400)
val_ds = TestDataset(800)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
trainer = Trainer(myloss, mymetric, optimizer, MODEL_NAME, model, None, DEVICE)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=WORKERS)

model.to(DEVICE)

for i in range(EPOCHS):
    trainer.train(train_loader)
    trainer.validate(val_loader)

trainer.writer.close()
trainer.writer.export_scalars_to_json("./all_scalars.json")
