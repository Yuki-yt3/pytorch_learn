from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "D:/pytorch_learn/dataset_read/dataset/train/bees/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
# 终端输入 tensorboard --logdir=tensorboard\logs --port=6007(设置端口为别的以免默认6006访问人数过多)
writer.close()