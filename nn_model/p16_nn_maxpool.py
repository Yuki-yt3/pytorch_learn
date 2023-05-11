import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("D:/pytorch_learn/tensorboard/dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)

# input = torch.reshape(input, (1, 1, 5, 5))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        self.maxpool2 = MaxPool2d(kernel_size=3, ceil_mode=False)


    def forward(self, input):
        output1 = self.maxpool1(input)
        output2 = self.maxpool2(input)
        return output1, output2

model = Model()
# output1, output2 = model(input)
# print(output1)
# print(output2)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output1, output2 = model(imgs)
    # print(imgs.shape)
    # print(output.shape)
    # torch.Size([16, 3, 32, 32])
    writer.add_images("inputs", imgs, step)

    # torch.Size([16, 6, 30, 30])->[xx, 3, 30, 30]
    writer.add_images("output", output1, step)
    step = step+1

writer.close()