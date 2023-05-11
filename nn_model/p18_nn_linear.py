import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
import torch.nn as nn

dataset = torchvision.datasets.CIFAR10("D:/pytorch_learn/tensorboard/dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

model = Model()
for data in dataloader:
    imgs, targets = data
    # print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    # print(output.shape)
    output = model(output)
    print(output.shape)