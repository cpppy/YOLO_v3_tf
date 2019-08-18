# coding: utf-8
# This script is modified from https://github.com/lars76/kmeans-anchor-boxes

from __future__ import division, print_function

import os

import numpy as np

import cv2


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def load_train_labels(img_dir, label_dir):
    label_fn_list = os.listdir(label_dir)
    bbox_size_list = []
    for i, label_fn in enumerate(label_fn_list):
        if i>60000:
            break
        label_fpath = os.path.join(label_dir, label_fn)
        img_fpath = os.path.join(img_dir, label_fn.replace('.txt', '.jpg'))
        img_cv2 = cv2.imread(img_fpath)
        h, w = img_cv2.shape[0:2]
        if h < w:
            new_h = 1280
            new_w = w * (new_h / h)
        else:
            new_w = 1280
            new_h = h * (new_w / w)
        new_h, new_w = 1280, 1280
        with open(label_fpath, 'r', encoding='utf-8') as f:
            obj_list = []
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                cls_idx = line.split(',')[-1]
                cls_name = line.split(',')[-2]
                bbox = line.split(',')[0:8]
                if len(bbox) < 8:
                    bbox = [bbox[0], bbox[1], bbox[0], bbox[3], bbox[2], bbox[3], bbox[2], bbox[1]]
                bbox = np.asarray(bbox, np.int)
                bbox = np.reshape(a=bbox, newshape=(4, 2))
                xmin, ymin = np.min(bbox[:, 0]), np.min(bbox[:, 1])
                xmax, ymax = np.max(bbox[:, 0]), np.max(bbox[:, 1])
                box_w = xmax - xmin
                box_h = ymax - ymin
                new_box_w = int(box_w * (new_w / w))
                new_box_h = int(box_h * (new_h / h))
                bbox_size_list.append([new_box_w, new_box_h])

    return np.asarray(bbox_size_list, dtype=np.int)


def get_kmeans(anno, cluster_num=9):
    anchors = kmeans(anno, cluster_num)
    ave_iou = avg_iou(anno, anchors)

    anchors = anchors.astype('int').tolist()

    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


#
# def updata_anchors():
#     annotation_path = params.train_data_list_path
#     anno_result = parse_annotations(annotation_path)
#     anchors, ave_iou = get_kmeans(anno_result, 9)
#
#     anchor_string = ''
#     for anchor in anchors:
#         anchor_string += '{},{}, '.format(anchor[0], anchor[1])
#     anchor_string = anchor_string[:-2]
#
#     with open(params.anchor_file_path, 'w', encoding='utf-8') as output_file:
#         output_file.writelines([anchor_string])
#
#     print('anchors are:')
#     print(anchor_string)
#     print('the average iou is:')
#     print(ave_iou)


if __name__ == '__main__':
    bbox_size_arr = load_train_labels(img_dir='/data/data/weche_train_data/images',
                                      label_dir='/data/data/weche_train_data/labels')
    print('boxes: ', bbox_size_arr.shape)

    anchors, ave_iou = get_kmeans(anno=bbox_size_arr, cluster_num=12)
    print('anchor: ', anchors)
    print('average_iou: ', ave_iou)
    #
    # anchor_string = ''
    # for anchor in anchors:
    #     anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    # anchor_string = anchor_string[:-2]
    #
    # print('anchors are:')
    # print(anchor_string)
    # print('the average iou is:')
    # print(ave_iou)
