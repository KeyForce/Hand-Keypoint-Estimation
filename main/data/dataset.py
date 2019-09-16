# -*- coding: utf-8 -*-
"""
@File    : dataset.py
@Time    : 2019/9/13 16:34
@Author  : KeyForce
@Email   : july.master@outlook.com
"""
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json


class ReadJsonPoint:
    """读取CMU手部21点关键点数据"""
    def __init__(self, json_path):
        self.json_path = json_path

    def read(self):
        hand_point = []

        with open(self.json_path, 'r') as f:
            hand_data = json.load(f)

        for i in range(21):
            # 这边要注意不要xy坐标搞混
            hand_tem_xy = hand_data['hand_pts'][i][:2]
            # hand_tem_xy.reverse()
            hand_point.append(hand_tem_xy)

        return hand_point


class CMUHandPointDataset(Dataset):
    """读取CMU手部关键点数据"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_name = []

        # 分离目录下的jpg和json
        file_list = os.listdir(root_dir)
        for i in file_list:
            if os.path.splitext(i)[1] == '.jpg':
                self.image_name.append(i)


    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.root_dir,
                                self.image_name[item])
        image = io.imread(img_path)
        json_path = os.path.join(img_path.replace('.jpg', '.json'))
        # 调用read方法读取数据
        landmarks = ReadJsonPoint(json_path).read()
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_name)


def show_landmarks(image, landmarks):
    """显示landmark，以方便检查数据"""
    plt.imshow(image)
    x = []
    y = []
    for i in range(21):
        x.append(landmarks[i][0])
        y.append(landmarks[i][1])
    plt.scatter(x, y, s=10, marker='.', c='r')
    plt.pause(0.001)


if __name__ == '__main__':
    root_dir = '/home/wild/Hand-Keypoint-Estimation/data/Hands from Synthetic Data (6546 + 3243 + 2348 ' \
               '+ 2124 = 14261 annotations)/hand_labels_synth/synth2'
    Data = CMUHandPointDataset(root_dir)

    fig = plt.figure()

    for i in range(len(Data)):
        sample = Data[i]

        print(i, sample['image'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break
