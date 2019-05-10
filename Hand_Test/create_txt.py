import os
import json
import cv2
import numpy

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

def get_json_point(json_path):
    hand_data_out = {}
    hand_return = {}
    str_point = ''
    cnt = numpy.zeros((21, 2), dtype=int)
    with open(json_path, 'r') as f:
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
    x, y, w, h = cv2.boundingRect(new_a)
    x, y, w, h = make_bbox_bigger([x, y, w, h], -0.08, -0.08, 0.8, 0.8)

    hand_return[0] = hand_data_out[1]
    hand_return[1] = hand_data_out[7]
    hand_return[2] = hand_data_out[11]
    hand_return[3] = hand_data_out[15]
    hand_return[4] = hand_data_out[19]

    # box
    hand_return[5] = [x, y]
    hand_return[6] = [w, h]
    for key, value in hand_return.items():
        for i in range(2):
            str_point += str(value[i])
            str_point += ' '

    return str_point


if __name__ == '__main__':
    data_sources = ['synth1', 'synth2', 'synth3', 'synth4']
    root_dir = '/home/wild/Hand-Keypoint-Estimation/Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)/hand_labels_synth'

    data = []

    for data_source in data_sources:
        im_dir = os.path.join(root_dir, data_source)
        for im_file in os.listdir(im_dir):
            if '.jpg' in im_file:
                name = im_file.rstrip('.jpg')
                json_file_path = os.path.join(root_dir, data_source, name + '.json')
                im_file_path = os.path.join(data_source, name + '.jpg')
                point = get_json_point(json_file_path)
                data.append(" ".join([im_file_path, point]))

    with open('{}/data.txt'.format(root_dir), 'w') as f:
            for image_point in data:
                f.write('{}\r\n'.format(image_point))

    train = data[:int(len(data) * 0.7)]
    test = data[int(len(data) * 0.7):]

    with open('{}/train.txt'.format(root_dir), 'w') as f:
            for image_point in data:
                f.write('{}\r\n'.format(image_point))

    with open('{}/test.txt'.format(root_dir), 'w') as f:
            for image_point in data:
                f.write('{}\r\n'.format(image_point))


# random.shuffle(test_data)
# random.shuffle(test_data)
# random.shuffle(train_data)
# random.shuffle(train_data)
#
# with open('test.txt', 'w') as f:
#     f.write('\n'.join(test_data))
# with open('trainval.txt', 'w') as f:
#     f.write('\n'.join(train_data))
