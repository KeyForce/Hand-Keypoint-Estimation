# -*- coding: utf-8 -*-
import json
from collections import OrderedDict
from torch.autograd import Variable

import torch
import torch.nn as nn
from fastai.vision import *
from fastai import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))


class handpose_model(nn.Module):
    def __init__(self):
        super().__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv6_2_CPM', 'Mconv7_stage2', 'Mconv7_stage3', \
                          'Mconv7_stage4', 'Mconv7_stage5', 'Mconv7_stage6']
        # stage 1
        block1_0 = OrderedDict({
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1_stage1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2_stage1': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3_stage1': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3': [512, 512, 3, 1, 1],
            'conv4_4': [512, 512, 3, 1, 1],
            'conv5_1': [512, 512, 3, 1, 1],
            'conv5_2': [512, 512, 3, 1, 1],
            'conv5_3_CPM': [512, 128, 3, 1, 1]})

        block1_1 = OrderedDict({
            'conv6_1_CPM': [128, 512, 1, 1, 0],
            'conv6_2_CPM': [512, 22, 1, 1, 0]
        })

        blocks = {}
        blocks['block1_0'] = block1_0
        blocks['block1_1'] = block1_1

        # stage 2-6
        for i in range(2, 7):
            blocks['block%d' % i] = OrderedDict({
                'Mconv1_stage%d' % i: [150, 128, 7, 1, 3],
                'Mconv2_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv3_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv4_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv5_stage%d' % i: [128, 128, 7, 1, 3],
                'Mconv6_stage%d' % i: [128, 128, 1, 1, 0],
                'Mconv7_stage%d' % i: [128, 22, 1, 1, 0]})

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_0 = blocks['block1_0']
        self.model1_1 = blocks['block1_1']
        self.model2 = blocks['block2']
        self.model3 = blocks['block3']
        self.model4 = blocks['block4']
        self.model5 = blocks['block5']
        self.model6 = blocks['block6']
        self.head_reg = nn.Sequential(
            Flatten(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(22*46*46, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 42),
            Reshape(-1, 21, 2),
            nn.Tanh())
        self._initialize_weights()

    def forward(self, x):
        out1_0 = self.model1_0(x)
        out1_1 = self.model1_1(out1_0)
        concat_stage2 = torch.cat([out1_1, out1_0], 1)
        out_stage2 = self.model2(concat_stage2)
        concat_stage3 = torch.cat([out_stage2, out1_0], 1)
        out_stage3 = self.model3(concat_stage3)
        concat_stage4 = torch.cat([out_stage3, out1_0], 1)
        out_stage4 = self.model4(concat_stage4)
        concat_stage5 = torch.cat([out_stage4, out1_0], 1)
        out_stage5 = self.model5(concat_stage5)
        concat_stage6 = torch.cat([out_stage5, out1_0], 1)
        out_stage6 = self.model6(concat_stage6)
        x = self.head_reg(out_stage6)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


image_path = '/home/hanwei-1/data/hand_labels_synth/synth2_3'


transforms = get_transforms(do_flip=False, max_zoom=1.05, max_warp=0.01,max_rotate=3, p_lighting=1)

def get_y_func(x):
    pre, ext = os.path.splitext(x)
    hand_data_out = []
    # pre = pre.replace('synth2', 'synth2_json')
    hand_data = json.load(open(pre + '.json'))
    for i in range(21):
        hand_tem_xy = hand_data['hand_pts'][i][:2]
        hand_tem_xy.reverse()
        hand_data_out.append(hand_tem_xy)
    return Tensor(hand_data_out)


data = (PointsItemList.from_folder(path=image_path, extensions=['.jpg'], presort=True)
        .split_by_rand_pct()
        .label_from_func(get_y_func)
        .transform(transforms, size=368, tfm_y=True, remove_out=False,
                   padding_mode='border', resize_method=ResizeMethod.PAD)
        .databunch(bs=32)
        .normalize(imagenet_stats))


class MSELossFlat(nn.MSELoss):
    def forward(self, input:Tensor, target:Tensor):
     return super().forward(input.view(-1), target.view(-1))


mse_loss_flat = MSELossFlat()


class L2Loss(torch.nn.Module):
    def __init__(self, batch_size):
        super(L2Loss, self).__init__()
        self.batch_size = batch_size

    def forward(self, x: Variable, y: Variable, weights: Variable = None):
        if weights is not None:
            val = (x-y) * weights[:x.data.shape[0], :, :, :] # Slice by shape[n,..] for batch size (last batch < batch_size)
        else:
            val = x-y
        l = torch.sum(val ** 2) / self.batch_size / 2
        return l


l2loss = L2Loss(batch_size=8)

net = handpose_model()


learn = Learner(data, net, loss_func=mse_loss_flat)
learn.fit_one_cycle(cyc_len=200, max_lr=0.0001)
learn.recorder.plot()
plt.show()
learn.lr_find()
learn.recorder.plot()
plt.show()
