import torch
import torchvision


# 定义训练设备
from PIL import Image
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_indict = ["Cat", "Dog"]
# 数据处理 resize和归一化
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

image_path = r"D:\2022机器学习\机器学习\数据\Cat_and_Dog\PetImages\test\Cat\12236.jpg"
image = Image.open(image_path)
plt.imshow(image)

image = transforms(image)

image = torch.reshape(image, (1, 3, 224, 224))
image = image.to(device)

model = torch.load("model_vgg16_1.pth").to(device)


model.eval()
with torch.no_grad():
    output = model(image)
    predict = torch.softmax(output, dim=1)
    proba, class_ind = torch.max(predict, 1)
print_res = "class: {}   prob: {:.3}".format(class_indict[class_ind],
                                                 proba.item())
plt.title(print_res)
plt.show()