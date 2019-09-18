# -*- coding: utf-8 -*-
"""
@File    : train.py
@Time    : 2019/9/13 16:33
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn


def Train(model, train_loader, criterion, optimizer, device, metrics=None, lr_scheduler=None, epoch=30):
    """
    训练模型
    :param model: 模型
    :param train_loader: 训练集
    :param criterion: 损失
    :param optimizer: 优化器
    :param device: GPU 或者CPU
    :param metrics: 评价指标
    :param lr_scheduler: 学习率调整
    :param epoch: 迭代次数
    :return:
    """
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        label = label.long()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        # Log
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch,
                         batch_idx * len(image),
                         len(train_loader.dataset),
                         100. * batch_idx / len(train_loader),
                         loss.item())
                  )


def Test(model, test_loader, criterion, device, epoch):
    """
    测试模型
    :param model: 模型
    :param test_loader: 测试集
    :param criterion: 损失
    :param device: GPU 或者CPU
    :param epoch:
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    confusion_matrix = np.zeros((21, 21))
    flag = 0
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            label = label.long()
            loss = criterion(output, label)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            # PA像素精度
            num_class = 21
            pre_image = pred.squeeze(1).cpu().numpy()

            gt_image = label.cpu().numpy()

            confusion_matrix = fast_hist(gt_image, pre_image, num_class)
            # plt.close()
            PA = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, PA: {}\n'.
                  format(loss,
                         PA,
                         )
                  )


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def main():
    # 使用drop_last让Batch能够整除
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, drop_last=True)

    # 设置GPU
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    # 初始化模型，损失，优化器
    model =
    loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8, weight_decay=5e-4)
    # 开始训练
    for epoch in range(40):
        Train(model, train_loader=train_loader,
              criterion=loss, optimizer=optimizer,
              device=device, epoch=epoch)
        Test(model, test_loader, loss, device, epoch)


if __name__ == '__main__':
    main()