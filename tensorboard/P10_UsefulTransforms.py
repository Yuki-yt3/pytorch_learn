from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("D:/pytorch_learn/dataset_read/dataset/train/ants/0013035.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize def __init__(self, mean, std, inplace=False):
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.3, 0.4, 0.5], [0.2, 0.6, 0.7])
img_norm = trans_norm(img_tensor)
# output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize #input:PIL image 等比缩放
print(img.size)
trans_resize = transforms.Resize((256, 256))
img_resize = trans_resize(img)
img_resize_totensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_totensor, 0)
print(img_resize.size)

# Compose -resize -2
trans_resize_2 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随机裁剪 #input PIL image
trans_random = transforms.RandomCrop((256, 400))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Random", img_crop, i)


writer.close()

