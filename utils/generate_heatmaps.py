import os
import json
import argparse
import numpy as np
from PIL import Image

'''
Most of the code borrowed from: 
https://github.com/lightas/ICCV19_Pose_Guided_Occluded_Person_ReID/blob/master/AlphaPose/generate_heatmap.py
'''

NUM_LANDMARKS = 18
HEATMAPS_SIZE = 25
LANDMARKS_THRESHOLD = 0.2

# Compute gaussian kernel


def _center_gaussian_heatmap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def _generate_pose_guided_heatmaps(cropped_targets_imgs_path, pose_data, save_path):
    for img in os.listdir(cropped_targets_imgs_path):
        img_ = Image.open(os.path.join(cropped_targets_imgs_path, img))
        img_np = np.array(img_)
        img_h, img_w = img_np.shape[0], img_np.shape[1]

        final_hmap = np.ones((NUM_LANDMARKS, HEATMAPS_SIZE, HEATMAPS_SIZE))

        img_id = img.split('.')[0]
        try:
            pose_landmarks = pose_data[img_id]
        except Exception as e:
            print(e, img_id)

        final_p = np.array(pose_landmarks)
        pose_landmarks = np.array(pose_landmarks)

        pose_landmarks = pose_landmarks[pose_landmarks[:, 2]
                                        > LANDMARKS_THRESHOLD]

        heatmap = np.zeros((NUM_LANDMARKS, img_h, img_w))
        l = final_p.shape[0]

        for j in range(l):
            if final_p[j].all() == 0.:
                heatmap[j] = np.ones((img_h, img_w))
            else:
                w, h = final_p[j][:2]
                w, h = int(w), int(h)
                heatmap[j] = _center_gaussian_heatmap(
                    img_h, img_w, w, h, (img_h*img_w/1000.))
        final_hmap = np.zeros((NUM_LANDMARKS, HEATMAPS_SIZE, HEATMAPS_SIZE))
        for j in range(18):
            b = Image.fromarray(heatmap[j]*255).convert('L')
            b = b.resize((HEATMAPS_SIZE, HEATMAPS_SIZE), Image.BILINEAR)
            b = np.array(b)
            final_hmap[j] = b

        np.save(os.path.join(save_path, img_id + '.npy'), final_hmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--cropped_targets_imgs_path', type=str,
                        default='', help='')
    parser.add_argument('--pose_data_path', type=str,
                        default='', help='')
    parser.add_argument('--save_path', type=str,
                        default='', help='')
    args = parser.parse_args()

    cropped_targets_imgs_path = "/home/anshu/TRAIN_DATA/targets"
    save_path = "/home/anshu/TRAIN_DATA/pose25x25"

    pose_data = None
    with open(args.pose_data_path) as data_file:
        pose_data = json.load(data_file)

    _generate_pose_guided_heatmaps(
        cropped_targets_imgs_path, pose_data, save_path)
