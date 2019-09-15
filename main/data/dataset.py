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

    def read_cmu(self):
        hand_point = []

        with open(self.json_path, 'r') as f:
            hand_data = json.load(f)

        for i in range(21):
            # 这边要注意不要xy坐标搞混
            hand_point_yx_to_xy = hand_data['hand_pts'][i][:2].reverse()
            hand_point.append(hand_point_yx_to_xy)

        return hand_point


class CMUHandPointDataset(Dataset):
    """读取CMU手部关键点数据"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_name = []
        self.json_name = []

        # 分离目录下的jpg和json
        file_list = os.listdir(root_dir)
        for i in file_list:
            if os.path.splitext(i)[1] == '.jpg':
                self.image_name.append(i)
                i.replace('.jpg', '.json')
                self.json_name.append(i)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.root_dir,
                                self.image_name[item])
        image = io.imread(img_path)
        json_path = os.path.join(self.root_dir, self.json_name[item])
        landmarks = ReadJsonPoint(json_path)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample




if __name__ == '__main__':
    root_dir = '/home/wild/Hand-Keypoint-Estimation/data/Hands from Synthetic Data (6546 + 3243 + 2348 ' \
               '+ 2124 = 14261 annotations)/hand_labels_synth/synth2'
    Data = CMUHandPointDataset(root_dir)
