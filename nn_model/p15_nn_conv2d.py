import torch
import torch.nn as nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("D:/pytorch_learn/tensorboard/dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = Model()
# print(model)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([16, 3, 32, 32])
    writer.add_images("inputs", imgs, step)

    # torch.Size([16, 6, 30, 30])->[xx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step+1

writer.close()