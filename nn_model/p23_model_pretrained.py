import torch
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split="train", download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn
from torch.utils.data import DataLoader

vgg16 = torchvision.models.vgg16(weights=None)
vgg16_true = torchvision.models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
# print("ok")
# print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("D:/pytorch_learn/tensorboard/dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

dataloader = DataLoader(train_data, batch_size=64)
# 添加一层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
# 修改最后一层
vgg16.classifier[6] = nn.Linear(4096, 10)
print(vgg16)
# 模型保存 method1，模型结构+模型参数
# torch.save(vgg16, "vgg16_method1.pth")
# method2，模型参数（官方推荐
# torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# 模型加载
model = torch.load("vgg16_method1.pth")
print(model)
# 模型修改过时需要先将修改的加入
vgg16_load = torchvision.models.vgg16(weights=None)
vgg16_load.classifier[6] = nn.Linear(4096, 10)
vgg16_load.load_state_dict(torch.load("vgg16_method2.pth"))
# model = torch.load("vgg16_method2.pth")
print(vgg16_load)
