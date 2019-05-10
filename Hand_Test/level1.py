#!/usr/bin/env python2.7
# coding: utf-8


import os
from os.path import join, exists

import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import getDataFromTxt
from utils import shuffle_in_unison_scary, logger, createDir, processImage

TRAIN = '/home/wild/Hand-Keypoint-Estimation/Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)/hand_labels_synth'
OUTPUT = '/home/wild/Face_Landmark/Hand_Test/Mytrain'
if not exists(OUTPUT):
    os.mkdir(OUTPUT)
assert(exists(TRAIN) and exists(OUTPUT))


def generate_hdf5(ftxt, output, fname, argument=False):

    data = getDataFromTxt(ftxt)
    F_imgs = []
    F_landmarks = []

    for (imgPath, landmarkGt, bbox) in data:
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        assert(img is not None)
        logger("process %s" % imgPath)
        # plt.imshow(img)
        # plt.show()

        f_face = img[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
        plt.imshow(f_face)
        plt.show()

        f_face = cv2.resize(f_face, (39, 39))

        f_face = f_face.reshape((1, 39, 39))

        f_landmark = landmarkGt.reshape((10))
        F_imgs.append(f_face)
        F_landmarks.append(f_landmark)



    F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)


    F_imgs = processImage(F_imgs)
    shuffle_in_unison_scary(F_imgs, F_landmarks)


    # full face
    base = join(OUTPUT, '1_F')
    createDir(base)
    output = join(base, fname)
    logger("generate %s" % output)


    with h5py.File(output, 'w') as h5:
        h5['data'] = F_imgs.astype(np.float32)
        h5['landmark'] = F_landmarks.astype(np.float32)



if __name__ == '__main__':

    h5_path = '/home/wild/Face_Landmark/Hand_Test/Mytrain'
    # 训练集
    train_txt = join(TRAIN, 'train.txt')
    generate_hdf5(train_txt, OUTPUT, 'train.h5', argument=True)
    # 测试集
    test_txt = join(TRAIN, 'test.txt')
    generate_hdf5(test_txt, OUTPUT, 'test.h5')

    with open(join(OUTPUT, '1_F/train.txt'), 'w') as fd:
        fd.write(h5_path+'/1_F/train.h5')

    with open(join(OUTPUT, '1_F/test.txt'), 'w') as fd:
        fd.write(h5_path+'/1_F/test.h5')

    print 'ok'