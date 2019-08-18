# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import cv2
import hsv_filter


def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed
    pic_path, boxes info, and label info.
    return:
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
    '''
    line = line.decode('utf-8')
    s = line.strip().split(' ')
    # print('s: ', s)
    pic_path = s[0]
    s = s[1:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return pic_path, boxes, labels


def resize_image_and_correct_boxes(img, boxes, img_size):
    # convert gray scale image to 3-channel fake RGB image
    if len(img) == 2:
        img = np.expand_dims(img, -1)
    ori_height, ori_width = img.shape[:2]
    new_width, new_height = img_size
    # shape to (new_height, new_width)
    img = cv2.resize(img, (new_width, new_height))

    # convert to float
    img = np.asarray(img, np.float32)

    # boxes
    # xmin, xmax
    boxes[:, 0] = boxes[:, 0] / ori_width * new_width
    boxes[:, 2] = boxes[:, 2] / ori_width * new_width
    # ymin, ymax
    boxes[:, 1] = boxes[:, 1] / ori_height * new_height
    boxes[:, 3] = boxes[:, 3] / ori_height * new_height

    return img, boxes


def data_augmentation(img, boxes, label):
    '''
    Do your own data augmentation here.
    param:
        img: a [H, W, 3] shape RGB format image, float32 dtype
        boxes: [N, 4] shape boxes coordinate info, N is the ground truth box number,
            4 elements in the second dimension are [x_min, y_min, x_max, y_max], float32 dtype
        label: [N] shape labels, int64 dtype (you should not convert to int32)
    '''
    return img, boxes, label


def process_box(boxes, labels, img_size, class_num, anchors):
    '''
    Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
    input:
    :param boxes: (N, 4), [x0, y0, x1, y1]
    :param labels: (,)
    :param img_size: (w, h)
    :param class_num: cls_num
    :param anchors: (9, 2), seq: box_size from small to large

    return:
    '''
    # (x_center, y_center)
    box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2  # ([x0,y0]+[x1,y1])/2
    # (width, height)
    box_sizes = boxes[:, 2:4] - boxes[:, 0:2]  # [w,h] = [x1,y1]-[x0,y0]

    # build y_true_maps
    img_w, img_h = img_size
    y_true_l = np.zeros(shape=(img_h // 8, img_w // 8, 3, 5 + class_num), dtype=np.float32)
    y_true_m = np.zeros(shape=(img_h // 16, img_w // 16, 3, 5 + class_num), dtype=np.float32)
    y_true_s = np.zeros(shape=(img_h // 32, img_w // 32, 3, 5 + class_num), dtype=np.float32)
    y_true = [y_true_l, y_true_m, y_true_s]

    # box_sizes: [N, 2] ---> [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, axis=1)
    # choose smaller half_side between each gt_box and each anchor
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    # distance from center_point to each union of one gt_box and one anchor, used to calc interaction_area of IOU
    left_top_points = np.maximum(-box_sizes / 2, -anchors / 2)
    right_bottom_points = np.minimum(box_sizes / 2, anchors / 2)
    # choose max_iou of 9 (in order to select best_match anchor for current gt_box)
    # interaction
    interaction_box_sizes = right_bottom_points - left_top_points  # [N, 9, 2]
    interaction_area = interaction_box_sizes[:, :, 0] * interaction_box_sizes[:, :, 1]  # [N, 9]
    # union: [N, 1] + [9] ---> [N, 9]
    union_area = box_sizes[:, :, 0]*box_sizes[:, :, 1] + anchors[:, 0]*anchors[:, 1]
    union_area = union_area - interaction_area
    iou = interaction_area / union_area

    best_match_anchor_idxes = np.argmax(iou, axis=1)  # [N]
    # draw label_map
    for gt_box_idx, anchor_idx in enumerate(best_match_anchor_idxes):
        # anchor_idx: (0, 1, 2)--->feature_map_l, (3, 4, 5)--->feature_map_m, (6, 7, 8)--->feature_map_s
        feature_map_idx = anchor_idx // 3
        downsample_ratio = [1/8, 1/16, 1/32][feature_map_idx]
        # find position of gt_box on feature_map
        x = int(np.floor(box_centers[gt_box_idx, 0]*downsample_ratio))
        y = int(np.floor(box_centers[gt_box_idx, 1]*downsample_ratio))
        anchor_idx_in_feature_map = anchor_idx % 3

        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, :2] = box_centers[gt_box_idx]   # center_point
        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, 2:4] = box_sizes[gt_box_idx]   # box_size: (w, h)
        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, 4] = 1.0                       # obj_proba
        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, 5+labels[gt_box_idx]] = 1.0    # cls_proba

    return y_true#[2], y_true[1], y_true[0]






def parse_data(line, class_num, img_size, anchors, mode):
    # print('line: ', line)
    '''
    param:
        line: a line from the training/test txt file
        args: args returned from the main program
        mode: 'train' or 'val'. When set to 'train', data_augmentation will be applied.
    '''
    pic_path, boxes, labels = parse_line(line)

    img = cv2.imread(pic_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, boxes = resize_image_and_correct_boxes(img, boxes, img_size)

    # do data augmentation here
    if mode == 'train':
        img, boxes, labels = data_augmentation(img, boxes, labels)

    # print('img_shape: ', img.shape)
    # the input of yolo_v3 should be in range 0~1
    # img = img / 255.
    img = hsv_filter.image_preprocess_by_normality(img, seq_type='RGB')

    y_true_s, y_true_m, y_true_l = process_box(boxes, labels, img_size, class_num, anchors)

    return img, y_true_s, y_true_m, y_true_l


if __name__ == '__main__':
    a = np.asarray([[[11, 2]], [[3, 4]]])
    print(a.shape)
    b = np.asarray([[3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    c = a+b
    print(c)
    print(c.shape)

