import json
import cv2
import numpy
import matplotlib.pyplot as plt

im_dir = '/home/wild/Hand-Keypoint-Estimation/Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)/hand_labels_synth/synth1/0001.jpg'
json_dir = '/home/wild/Hand-Keypoint-Estimation/Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)/hand_labels_synth/synth1/0001.json'
hand_data_out = {}

hand_data_out = {}
cnt = numpy.zeros((21, 2), dtype=int)
with open(json_dir, 'r') as f:
    hand_data = json.load(f)

for i in range(21):
    hand_data_out[i] = hand_data['hand_pts'][i][:2]

for j in range(21):
    for i in range(2):
        hand_data_out[j][i] = int(hand_data_out[j][i])

for i in range(21):
    cnt[i] = numpy.array(hand_data_out[i])

index = [4, 8, 12, 16, 20]
new_a = numpy.delete(cnt, index, axis=0)
img = cv2.imread(im_dir)
x, y, w, h = cv2.boundingRect(new_a)


def make_bbox_bigger(data, xR, yR, wR, hR):

    xDelta = data[0] * xR
    yDelta = data[1] * yR
    wDelta = data[2] * wR
    hDelta = data[3] * hR

    x = data[0] + xDelta
    y = data[1] + yDelta
    w = data[2] + wDelta
    h = data[3] + hDelta
    return [int(x), int(y), int(w), int(h)]

x, y, w, h = make_bbox_bigger([x, y, w, h], -0.08, -0.08, 0.08, 0.08)

cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imwrite('hand.jpeg', img)
plt.imshow(img)
plt.show()


