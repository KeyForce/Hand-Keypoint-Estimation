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
        self.hand_point = []

    def read(self):
        with open(self.json_path, 'r') as f:
            hand_data = json.load(f)

        for i in range(21):
            # 这边要注意不要xy坐标搞混
            hand_tem_xy = hand_data['hand_pts'][i][:2]
            hand_tem_xy = list(map(int, hand_tem_xy))
            self.hand_point.append(hand_tem_xy)

        # hand_point = list(map(int, hand_point))

        return np.array(self.hand_point)


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


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]


        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


def show_landmarks(image, landmarks):
    """显示landmark，以方便检查数据"""
    plt.imshow(image)
    x = []
    y = []
    for i in range(21):
        x.append(landmarks[i][0])
        y.append(landmarks[i][1])
    plt.scatter(x, y, s=10, marker='.', c='r')


if __name__ == '__main__':
    root_dir = '/home/wild/Hand-Keypoint-Estimation/data/Hands from Synthetic Data (6546 + 3243 + 2348 ' \
               '+ 2124 = 14261 annotations)/hand_labels_synth/synth2'

    composed = transforms.Compose([Rescale(368),
                                   ToTensor()])

    Data = CMUHandPointDataset(root_dir, composed)

    for i in range(8):
        sample = Data[i]

        print(i, sample['image'].shape)
        print('First 4 Landmarks: {}'.format(sample['landmarks'][:4]))
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(sample['image'].permute(1, 2, 0))
        x = []
        y = []
        for i in range(21):
            x.append(np.array(sample['landmarks'][i][0]))
            y.append(np.array(sample['landmarks'][i][1]))
        plt.scatter(x, y, s=10, marker='.', c='r')

    plt.show()
