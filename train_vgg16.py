import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss, Sequential, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    # 数据处理
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

    # 网络模型
    model_vgg16 = torchvision.models.vgg16(pretrained=True)
    # # 修改网络结构，将fc层1000个输出改为2个输出
    # dim_in = model_vgg16.fc.in_features
    # model_vgg16.fc = nn.Linear(dim_in, 2)
    model_vgg16.add_module("linear", Linear(1000, 2))
    # print(model_vgg16)
    # 使用gpu训练
    model_vgg16.cuda()

    # 损失函数
    loss_fn = CrossEntropyLoss()
    # 使用gpu训练
    loss_fn = loss_fn.cuda()

    # 优化器
    learn_rate = 0.0001
    optim = torch.optim.Adam(model_vgg16.parameters(), lr=learn_rate)

    writer = SummaryWriter("logs_train")

    epoch = 10
    for i in range(epoch):
        print("**************第{}轮训练开始**************".format(i+1))
        # 记录模型训练损失、准确数
        train_loss = 0
        train_accuracy = 0
        # 训练状态
        # enumerate()会返回两个值, 一个是索引, 一个是数据
        for batch_idx, data in enumerate(train_dataloader, 1):
            # 模型进入训练状态
            model_vgg16.train()
            imgs, targets = data
            imgs, targets = imgs.cuda(), targets.cuda()

            output = model_vgg16(imgs)
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

        # 验证状态
        # 记录验证损失、准确数
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_dataloader, 1):
                # 模型进入验证状态
                model_vgg16.eval()
                imgs, targets = data
                imgs, targets = imgs.cuda(), targets.cuda()

                output = model_vgg16(imgs)
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
    torch.save(model_vgg16, "model_vgg16_1.pth")

    writer.close()


    # # 设置训练网络的参数
    # # 记录训练的次数
    # total_train = 0
    # total_test = 0
    # # 记录训练的轮次
    # epoch = 10
    #
    # writer = SummaryWriter("logs_train")
    #
    #
    # for i in range(epoch):
    #     # 模型训练
    #     # 网络模型进入训练状态
    #     model_vgg16.train()
    #     for data in train_dataloader:
    #         imgs, targets = data
    #         # 使用gpu训练
    #         imgs =imgs.cuda()
    #         targets = targets.cuda()
    #
    #         outputs = model_vgg16(imgs)
    #         loss = loss_fn(outputs, targets)
    #
    #         # 优化器模型优化
    #         optim.zero_grad()
    #         loss.backward()
    #         optim.step()
    #         total_train = total_train + 1
    #         if total_train % 100 == 0:
    #             print("训练次数{}，Loss：{}".format(total_train, loss.item()))
    #             writer.add_scalar("train_loss", loss.item(), total_train)
    #
    #     # 模型测试
    #     # 模型进入测试模式
    #     model_vgg16.eval()
    #     # 总损失
    #     total_test_loss = 0
    #     # 总正确数
    #     total_accuracy = 0
    #     with torch.no_grad():
    #         for data in test_dataloader:
    #             imgs, targets = data
    #             # 使用gpu训练
    #             imgs = imgs.cuda()
    #             targets = targets.cuda()
    #             outputs = model_vgg16(imgs)
    #             loss = loss_fn(outputs, targets)
    #             # 记录总损失
    #             total_test_loss = total_test_loss + loss.item()
    #             # 记录测试准确数
    #             accuracy = (outputs.argmax(1) == targets).sum()
    #             # 记录测试准确数量总和
    #             total_accuracy = total_accuracy + accuracy
    #     print("整体测试集上的损失：{}".format(total_test_loss))
    #     print("整体测试集上准确率：{}%".format((100. * total_accuracy) / test_data_len))
    #     writer.add_scalar("test_loss", total_test_loss, total_test)
    #     writer.add_scalar("test_accuracy", total_accuracy / test_data_len, total_test)
    #     total_test = total_test + 1
    #     if i == 9:
    #         torch.save(model_vgg16, "model_vgg16_{}.pth".format(i))
    # writer.close()
    # 使用tensorboard中的SummaryWriter类绘制模型训练loss_acc图像