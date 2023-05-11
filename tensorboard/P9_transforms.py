from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法-》tensor数据类型
# 通过transforms.toTensor去看两个问题

# 2、为什么需要Tensor数据类型

img_path = "D:/pytorch_learn/dataset_read/dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("logs")

# 1、transforms该如何使用
tensor_trans = transforms.ToTensor()  # 创建具体的工具
tensor_img = tensor_trans(img)
# print(tensor_img)

writer.add_image("Tensor_image", tensor_img)

writer.close()