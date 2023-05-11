import torch
import torch.nn as nn
import torchvision
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

dataset = torchvision.datasets.CIFAR10("D:/pytorch_learn/tensorboard/dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        # output = self.relu1(input)
        output = self.sigmoid1(input)
        return output


model = Model()
# output = model(input)
# print(output)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
