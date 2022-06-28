import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
if __name__ == '__main__':

    # 定义训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据处理 resize和归一化
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_dataset = torchvision.datasets.ImageFolder(root=r"D:\2022机器学习\机器学习\数据\Cat_and_Dog\PetImages\train", transform=transforms)
    print(train_dataset.class_to_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torchvision.datasets.ImageFolder(root=r"D:\2022机器学习\机器学习\数据\Cat_and_Dog\PetImages\val", transform=transforms)
    print(test_dataset.class_to_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 训练数据长度
    train_data_len = len(train_dataset)
    test_data_len = len(test_dataset)

    # 迁移学习模型
    model_squeezenet1_0 = models.squeezenet1_0(pretrained=True)

    # 添加全连接层将输出改为2
    model_squeezenet1_0.add_module("linear", Linear(1000, 2))
    model_squeezenet1_0.to(device)

    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    # 优化器
    optim = torch.optim.Adam(model_squeezenet1_0.parameters(), lr=0.00001)

    # 使用tensorboard中的SummaryWriter类绘制模型训练loss_acc图像
    writer = SummaryWriter("logs_train_model_squeezenet1_0")

    epoch = 20
    for i in range(epoch):
        print("**************第{}轮训练开始**************".format(i+1))
        # 记录模型训练损失、准确数
        train_loss = 0
        train_accuracy = 0
        # 训练状态
        # enumerate()会返回两个值, 一个是索引, 一个是数据
        for batch_idx, data in enumerate(train_dataloader, 1):
            # 模型进入训练状态
            model_squeezenet1_0.train()
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)

            output = model_squeezenet1_0(imgs)
            # torch.max()函数
            # input是softmax函数输出的一个tensor
            # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
            # 会返回两个tensor，第一个tensor是每行的最大值，第二个tensor是每行最大值的索引
            _, pred = torch.max(output.data, 1)
            loss = loss_fn(output, targets)
            # 模型优化
            optim.zero_grad()
            loss.backward()
            optim.step()
            # 记录总损失 总准确数
            train_loss += loss.item()
            train_accuracy += torch.sum(pred == targets)

            if batch_idx % 250 == 0:
                print("Batch:{},Train Loss:{:.4f},Train ACC:{:.4f}".format(batch_idx, train_loss / batch_idx,
                                                                          100 * train_accuracy / len(train_dataset)))
        writer.add_scalar("train_loss_epoch", train_loss, i)
        writer.add_scalar("train_accuracy_epoch", train_accuracy / len(train_dataset), i)
        # epoch_loss = train_loss  / len(train_dataset)
        # epoch_acc = 100 * train_accuracy / len(train_dataset)
        # print("train Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))

        # 验证状态
        # 记录验证损失、准确数
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader, 1):
                # 模型进入验证状态
                model_squeezenet1_0.eval()
                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                output = model_squeezenet1_0(imgs)
                # torch.max()这个函数返回的是两个值：一个值是具体的value 一个值是value所在的index
                _, pred = torch.max(output.data, 1)
                loss = loss_fn(output, targets)
                # optim.zero_grad()

                test_loss += loss.item()
                test_accuracy += torch.sum(pred == targets)
        epoch_loss = test_loss / len(test_dataset)
        epoch_acc = 100 * test_accuracy / len(test_dataset)
        print("test Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss, epoch_acc))
        writer.add_scalar("test_loss_epoch", test_loss, i)
        writer.add_scalar("test_accuracy_epoch", test_accuracy / len(test_dataset), i)
    # 保存模型
    torch.save(model_squeezenet1_0, "model_squeezenet1_0_1.pth")

    writer.close()