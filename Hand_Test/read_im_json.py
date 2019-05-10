import json
import cv2
im_dir = '/home/wild/Hand-Keypoint-Estimation/Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)/hand_labels_synth/synth1/0001.jpg'
json_dir = '/home/wild/Hand-Keypoint-Estimation/Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)/hand_labels_synth/synth1/0001.json'
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

    return str_point

data = get_json_point(json_dir)

# output = cv2.imread(im_dir)
# for i in range(5):
#     cv2.circle(output, tuple(data[i]), 2, (0, 0, 255), 1)
# cv2.imshow("capture", output)
# while True:
#     if cv2.waitKey(1) == 27:
#         break  # esc to quit