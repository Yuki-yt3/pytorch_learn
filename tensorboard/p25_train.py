import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from torch.utils.data import DataLoader
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的大小为：{}".format(train_data_size), ",测试数据集的大小为:{}".format(test_data_size))

# 加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 创建模型
model = Model()


# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learningrate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learningrate)

# 设置训练参数
total_train_step = 0
total_test_step = 0
epoch = 20

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()

for i in range(epoch):
    print("----------第{}轮训练开始---------".format(i + 1))
    # 训练
    model.train()
    for data in train_data_loader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{}，loss：{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # torch.save(model, "model_{}.pth".format(i+1))

writer.close()