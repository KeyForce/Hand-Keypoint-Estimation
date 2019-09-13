# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import cv2

im_dir = 'Path'
json_dir = 'Path'
hand_data_out = {}

with open(json_dir, 'r') as f:
    hand_data = json.load(f)

for i in range(21):
    hand_data_out[i] = hand_data['hand_pts'][i][:2]

for j in range(21):
    for i in range(2):
        hand_data_out[j][i] = int(hand_data_out[j][i])


def get_json_point(json_path):
    hand_data_out = {}
    hand_return = {}
    str_point = ''
    with open(json_dir, 'r') as f:
        hand_data = json.load(f)

    for i in range(21):
        hand_data_out[i] = hand_data['hand_pts'][i][:2]

    for j in range(21):
        for i in range(2):
            hand_data_out[j][i] = int(hand_data_out[j][i])

    hand_return[0] = hand_data_out[1]
    hand_return[1] = hand_data_out[7]
    hand_return[2] = hand_data_out[11]
    hand_return[3] = hand_data_out[15]
    hand_return[4] = hand_data_out[19]
    for key, value in hand_return.items():
        for i in range(2):
            str_point += str(value[i])
            str_point += ' '

    return hand_data_out


data = get_json_point(json_dir)

output = cv2.imread(im_dir)
for i in range(21):
    cv2.circle(output, tuple(data[i]), 2, (0, 0, 255), 1)
plt.imshow(output)
plt.show()

