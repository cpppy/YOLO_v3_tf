import numpy as np
import cv2
import os
import sys
sys.path.append('../')

import config


def is_valid_cord(x, y, w, h):
    return x >= 0 and x < w and y >= 0 and y < h


def get_neighbours_8(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
            (x - 1, y), (x + 1, y),
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_rot_rect_from_vertexes(bbox_xys):
    box = np.reshape(bbox_xys, newshape=[4, 2]).astype(np.int)
    rect = cv2.minAreaRect(box)
    x, y, w, h, theta = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    rot_rect = [x, y, w, h, theta]  # [x, y, w, h, theta]
    return rot_rect


def encode_single_image(normed_xs, normed_ys, labels):
    """
    Args:
        xs, ys: (N, 4),
        labels: shape = (N,)
             -1: ignored
              1: text
    Return:
        cls_label, cls_weight: (H, W, 1)
        link_label, link_weight: (H, W, 8)
        rects_with_labels:(N, 6) each one is [x_c, y_c, w, h, theta, label]
    """
    # validate the args
    assert np.ndim(normed_xs) == 2
    assert np.shape(normed_xs)[-1] == 4
    assert np.shape(normed_xs) == np.shape(normed_ys)
    assert len(normed_xs) == len(labels)

    W, H = config.img_w, config.img_h
    xs = normed_xs * W
    ys = normed_ys * H
    cls_label = np.zeros([H, W, 1], dtype=np.int32)
    cls_weight = np.zeros([H, W, 1], dtype=np.float32)
    link_label = np.zeros((H, W, 8), dtype=np.int32)
    link_weight = np.ones((H, W, 8), dtype=np.float32)

    num_positive_pixels_arr = []

    bbox_masks = []
    border_masks = []
    mask = np.zeros([H, W], dtype=np.int32)
    rects_with_labels = []
    # for cls, num in label_num_map.items():
    # if num <= 0:  # background
    #     continue
    for bbox_idx, (bbox_xs, bbox_ys, label) in enumerate(list(zip(xs, ys, labels))):
        # print('---------------', bbox_idx)
        # print('bbox_xs: ', bbox_xs)
        # print('bbox_ys: ', bbox_ys)
        bbox_xs, bbox_ys = np.clip(bbox_xs, 0, W), np.clip(bbox_ys, 0, H)
        bbox_xys = np.stack((bbox_xs, bbox_ys), axis=1)
        rot_rect = get_rot_rect_from_vertexes(bbox_xys)  # [x,y,w,h,theta]
        rot_rect.append(label)
        rects_with_labels.append(rot_rect)  # [N=1, [x,y,w,h,theta,label=1]]

        each_bbox_mask = cv2.fillPoly(mask.copy(), [bbox_xys.astype(np.int)], 1)
        # cv2.imshow('test', np.asarray(mask*255, dtype='uint8'))
        # cv2.waitKey(0)

        MINIMUM_MASK_REMAINED = 20
        if np.sum(each_bbox_mask) <= MINIMUM_MASK_REMAINED:
            continue

        bbox_masks.append(each_bbox_mask)
        num_positive_pixels_arr.append(np.sum(each_bbox_mask))
        each_bbox_edge_mask = cv2.polylines(img=mask.copy(),
                                            pts=[bbox_xys.astype(np.int)],
                                            isClosed=True,
                                            color=1,
                                            thickness=1)

        # cv2.imshow('test', np.asarray(bbox_border_mask_each_bbox*255, dtype='uint8'))
        # cv2.waitKey(0)

        border_masks.append(each_bbox_edge_mask)

        # 4: encode cls_label and cls_weight
        cls_label[:, :, 0] = np.clip(
            np.where(each_bbox_mask, cls_label[:, :, 0] + label, cls_label[:, :, 0]),
            a_min=0,
            a_max=label)

        # 5: encode link_label and link_weight
        bbox_cls_cords = np.where(each_bbox_mask)
        link_label[bbox_cls_cords] = 1
        bbox_border_cords = np.where(each_bbox_edge_mask)
        border_points = list(zip(*bbox_border_cords))

        for y, x in border_points:
            neighbours = get_neighbours_8(x, y)
            for n_idx, (nx, ny) in enumerate(neighbours):
                if not is_valid_cord(nx, ny, W, H) or not each_bbox_mask[nx, ny]:
                    link_label[y, x, n_idx] = 0


    # calc cls_weight
    # METHOD: instance balanced cross entropy loss
    # print('num_pos_arr: ', num_positive_pixels_arr)
    num_positive_pixels_arr = np.sum(a=bbox_masks, axis=(1, 2))
    sum_positive_pixels = np.sum(num_positive_pixels_arr)
    per_bbox_weight = sum_positive_pixels / len(num_positive_pixels_arr)
    # print(per_bbox_weight)
    per_pixel_weight_arr = per_bbox_weight / num_positive_pixels_arr
    # print(per_pixel_weight_arr)
    weighted_bbox_masks = [each_weight*each_mask for each_weight, each_mask in zip(per_pixel_weight_arr, bbox_masks)]
    # print(np.shape(weighted_bbox_masks))
    cls_weight = np.sum(weighted_bbox_masks, axis=0)

    cls_weight = np.expand_dims(cls_weight, axis=-1)  #(640, 640, 1)


    # num_positive_bboxes_each_cls = Counter(labels)
    # num_positive_pixels_each_cls = np.sum(cls_label, axis=2)
    # num_total_positive_pixels = np.sum(cls_label)
    # overlap area will greater then 1
    # WEIGHT_THRESHOLD = 200
    # cls_weight = np.clip(cls_weight, a_min=0, a_max=WEIGHT_THRESHOLD)


    # cls_weight = np.where((cls_label > 0), 1, 0).astype(np.float)

    link_weight = link_weight * cls_weight
    # print(link_weight.shape)

    # cls_label.astype(np.int32)
    cls_label = np.cast["int32"](cls_label)
    cls_weight = cls_weight.astype(np.float32)
    # cls_weight = np.cast["float32"](cls_weight)
    # link_label.astype(np.int32)
    link_label = np.cast["int32"](link_label)
    # link_weight.astype(np.float32)
    link_weight = np.cast["float32"](link_weight)
    # rects_with_labels = np.asarray(rects_with_labels, dtype=np.float32)
    rects_with_labels = np.cast["float32"](rects_with_labels)

    return cls_label, cls_weight, link_label, link_weight, rects_with_labels


def parse_labels(label_path):
    with open(os.path.join(label_path), 'r', encoding="utf-8") as f:
        lines = f.readlines()
        bbox_list = []
        for idx, line in enumerate(lines):
            elem_list = lines[idx].strip().split(',')[:8]
            coords = [float(num) for num in elem_list]
            bbox_list.append(coords)
        return bbox_list


def draw_mask(img_shape, bbox_list):
    mask = np.zeros(shape=img_shape[0:2])
    for bbox in bbox_list:
        vertexes = np.reshape(np.asarray(bbox, dtype=np.int), (4, 2))
        cv2.fillPoly(img=mask, pts=[vertexes], color=255)
    mask = cv2.resize(mask, dsize=(config.img_w, config.img_h))
    return mask


if __name__ == '__main__':

    img_path = './1_frontage_2.jpg'
    label_path = img_path.replace('.jpg', '.txt')

    img_cv2 = cv2.imread(img_path)
    bbox_list = parse_labels(label_path)
    mask = draw_mask(img_cv2.shape, bbox_list)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    h, w = img_cv2.shape[0:2]
    normed_xs = []
    normed_ys = []
    labels = []
    for bbox in bbox_list:
        vertexes = np.reshape(np.asarray(bbox, dtype=np.float), (4, 2))
        xs = vertexes[:, 0]
        normed_xs.append(xs / w)
        ys = vertexes[:, 1]
        normed_ys.append(ys / h)
        labels.append(1)
        # break
        # print(np.shape(normed_xs))
        # print(np.asarray([xs, xs]).shape)
        # xys = np.stack(np.asarray([xs, xs]), axis=1)
        # print('xys: ', xys)
        # exit(0)
    normed_xs = np.asarray(normed_xs)
    normed_ys = np.asarray(normed_ys)
    labels = np.asarray(labels)

    cls_label, cls_weight, link_label, link_weight, rects_with_labels = encode_single_image(normed_xs, normed_ys,
                                                                                            labels)
    # cls_label = np.squeeze(cls_label, axis=2)
    print(cls_label.shape)
    print(cls_weight.shape)
    print(link_label.shape)
    print(link_weight.shape)

    hot_map = cv2.applyColorMap(np.uint8(255 * link_weight[:, :, 0]), cv2.COLORMAP_JET)
    cv2.imshow('hot_map', hot_map)
    cv2.waitKey(0)
    print(rects_with_labels.shape)

    # print(cls_weight.shape)
