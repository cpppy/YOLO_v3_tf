import tensorflow as tf
import os
import numpy as np
import cv2

import sys

sys.path.append('../')

import config


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
    union_area = box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1]
    union_area = union_area - interaction_area
    iou = interaction_area / union_area

    best_match_anchor_idxes = np.argmax(iou, axis=1)  # [N]
    # draw label_map
    for gt_box_idx, anchor_idx in enumerate(best_match_anchor_idxes):
        # anchor_idx: (0, 1, 2)--->feature_map_l, (3, 4, 5)--->feature_map_m, (6, 7, 8)--->feature_map_s
        feature_map_idx = anchor_idx // 3
        downsample_ratio = [1 / 8, 1 / 16, 1 / 32][feature_map_idx]
        # find position of gt_box on feature_map
        x = int(np.floor(box_centers[gt_box_idx, 0] * downsample_ratio))
        y = int(np.floor(box_centers[gt_box_idx, 1] * downsample_ratio))
        anchor_idx_in_feature_map = anchor_idx % 3

        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, :2] = box_centers[gt_box_idx]  # center_point
        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, 2:4] = box_sizes[gt_box_idx]  # box_size: (w, h)
        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, 4] = 1.0  # obj_proba
        y_true[feature_map_idx][y, x, anchor_idx_in_feature_map, 5 + labels[gt_box_idx]] = 1.0  # cls_proba

    return y_true  # [2], y_true[1], y_true[0]



def tf_image_whitening(img_tensor):

    # mean_RGB = [30.89, 30.01, 32.26]
    # std = 37.76
    img_tensor = img_tensor-config.RGB_MEAN
    img_tensor = img_tensor/config.STD
    return img_tensor

class Read_Tfrecord:

    def __init__(self, tfrecord_dir, anchors, subset='train'):
        self.tfrecord_dir = tfrecord_dir
        self.anchors = anchors
        self.subset = subset

    def parser(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encode': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
                'label/x1': tf.VarLenFeature(dtype=tf.float32),
                'label/x2': tf.VarLenFeature(dtype=tf.float32),
                'label/x3': tf.VarLenFeature(dtype=tf.float32),
                'label/x4': tf.VarLenFeature(dtype=tf.float32),
                'label/y1': tf.VarLenFeature(dtype=tf.float32),
                'label/y2': tf.VarLenFeature(dtype=tf.float32),
                'label/y3': tf.VarLenFeature(dtype=tf.float32),
                'label/y4': tf.VarLenFeature(dtype=tf.float32),
                'label/cls': tf.VarLenFeature(dtype=tf.int64),
            }
        )

        # image = tf.decode_raw(, tf.uint8)
        # image = tf.cast(image, tf.uint8)
        image = tf.image.decode_jpeg(contents=features['image/encode'], channels=3)
        # image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
        # image = tf.identity(input=image)
        print('image_shape: ', image.get_shape().as_list())
        image = tf.image.resize_bilinear([image], size=[config.img_h, config.img_w])[0]
        # image = tf.reshape(tensor=image, shape=[config.img_h, config.img_w, config.img_ch])
        # print('image_shape: ', image.get_shape().as_list())
        image = tf.cast(image, dtype=tf.float32)
        image = tf_image_whitening(image)

        cls_idxes = tf.cast(features['label/cls'], dtype=tf.float32)
        # cls_idxes = tf.sparse.reset_shape(sp_input=cls_idxes, new_shape=[config.cls_num])
        # cls_idxes = tf.sparse.to_dense(sp_input=cls_idxes,
        #                                default_value=0,
        #                                validate_indices=False)

        label = tf.stack([tf.sparse.to_dense(features['label/x1']),
                          tf.sparse.to_dense(features['label/y1']),
                          tf.sparse.to_dense(features['label/x3']),
                          tf.sparse.to_dense(features['label/y3']),
                          tf.sparse.to_dense(cls_idxes),
                          ], axis=1)

        paddings = [[0, 100 - tf.shape(label)[0]], [0, 0]]
        label = tf.pad(label, paddings)

        return image, label

    def make_batch(self, batch_size):
        tfrecord_fn_list = [fn for fn in os.listdir(self.tfrecord_dir) if self.subset in fn]
        tfrecord_fpath_list = [os.path.join(self.tfrecord_dir, fn) for fn in tfrecord_fn_list]

        dataset = tf.data.TFRecordDataset(filenames=tfrecord_fpath_list)
        dataset = dataset.map(self.parser, num_parallel_calls=1)

        # batch it up
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        # transfer labels to model_output_structure
        label_batch = transform_targets(y_train=label_batch, anchors=self.anchors)
        return image_batch, label_batch


def transform_targets(y_train, anchors):
    '''

    :param y_train: [N, 100, 5]
    :param anchors: [M, 2], (w, h)
    :param anchor_masks: [[0,1,2],[3,4,5],[6,7,8]]
    :param classes: [N, 129]
    :return:
    '''

    # select best_match_anchor for each target box
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    # print(y_train.get_shape().as_list())

    recovery_ratio = tf.constant([config.img_w, config.img_h, config.img_w, config.img_h, 1], dtype=tf.float32)
    y_train = y_train * recovery_ratio
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                   tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)
    # return y_train

    y_outs = tf.py_func(coordinates_to_model_output, [y_train], [tf.float32, tf.float32, tf.float32])
    for i in range(3):
        grid_size = [config.img_h//8, config.img_h//16, config.img_h//32][i]
        y_outs[i].set_shape([None, grid_size, grid_size, 3, 5+config.cls_num])

    return y_outs


def coordinates_to_model_output(y_train):
    '''
    :param y_train: [N, M, 6]
        N: batch_size
        M: bboxes_num, padding to 100
        6: [x1, y1, x2, y2, cls_idx, best_anchor_idx]
    :param grid_size: 52/26/13
    :param anchor_idxs: (0,1,2)/(3,4,5)/(6,7,8)
    :param n_class: 129
    :return: y_outs: tuple([N, grid, grid, 3, (4+1+n_cls)], [], [])
    '''
    batch_size = y_train.shape[0]
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # grid_size_masks = [52, 26, 13]
    grid_size_arr = [config.img_h // 8, config.img_h // 16, config.img_h // 32]
    # ratio_dict = {1.: 8., 2.: 16., 3.: 32.}

    y_out_52 = np.zeros([batch_size, grid_size_arr[0], grid_size_arr[0], 3, 5 + config.cls_num], dtype=np.float32)
    y_out_26 = np.zeros([batch_size, grid_size_arr[1], grid_size_arr[1], 3, 5 + config.cls_num], dtype=np.float32)
    y_out_13 = np.zeros([batch_size, grid_size_arr[2], grid_size_arr[2], 3, 5 + config.cls_num], dtype=np.float32)

    y_true = [y_out_52, y_out_26, y_out_13]
    # print(y_train.shape)
    for bs_i in range(y_train.shape[0]):
        for box_i in range(y_train.shape[1]):
            if y_train[bs_i][box_i][2] < 1e-3:
                continue
            box = y_train[bs_i][box_i][0:4]
            box_center = (box[0:2] + box[2:4]) / 2
            box_size = box[2:4] - box[0:2] # (w,h)
            cls_idx = int(y_train[bs_i][box_i][4])
            best_match_anchor_idx = int(y_train[bs_i][box_i][5])
            feature_map_idx = 2 - best_match_anchor_idx // 3

            downsample_ratio = [1 / 8, 1 / 16, 1 / 32][feature_map_idx]
            # find position of gt_box on feature_map
            x = int(np.floor(box_center[0] * downsample_ratio))
            y = int(np.floor(box_center[1] * downsample_ratio))
            anchor_pos = best_match_anchor_idx % 3

            y_true[feature_map_idx][bs_i, y, x, anchor_pos, :2] = box_center  # center_point
            y_true[feature_map_idx][bs_i, y, x, anchor_pos, 2:4] = box_size  # box_size: (w, h)
            y_true[feature_map_idx][bs_i, y, x, anchor_pos, 4] = 1.0  # obj_proba
            y_true[feature_map_idx][bs_i, y, x, anchor_pos, 5 + cls_idx] = 1.0  # cls_proba

    return y_true


def transform_targets_for_output(y_train, grid_size, anchor_idxs, n_class=129):
    '''
    :param y_true: [N, M, 6]
        N: batch_size
        M: bboxes_num, padding to 100
        6: [x1, y1, x2, y2, cls_idx, best_anchor_idx]
    :param grid_size: 52/26/13
    :param anchor_idxs:(0,1,2)/(3,4,5)/(6,7,8)
    :param n_class:129
    :return:
    '''

    y_outs = tf.py_func(coordinates_to_model_output, [y_train], [tf.float32])

    return y_outs

    '''
    # N = y_true.get_shape().as_list()[0]
    N = config.batch_size
    # y_true_out: (N, grid, grid, n_anchor, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(shape=[N, grid_size, grid_size, len(anchor_idxs), n_class], dtype=tf.float32)
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    print(anchor_idxs.get_shape().as_list())

    # indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    # updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for bs_idx in range(N):
        for box_idx in range(100):
            condition =  tf.less(y_true[bs_idx][box_idx][2], 1e-3)
            def transform():
                print(box_idx)
                anchor_eq = tf.equal(
                    anchor_idxs, tf.cast(y_true[bs_idx][box_idx][5], tf.int32))

                if tf.reduce_any(anchor_eq):
                    box = y_true[bs_idx][box_idx][0:4]
                    # box_xy ---> center_point
                    box_xy = (y_true[bs_idx][box_idx][0:2] + y_true[bs_idx][box_idx][2:4]) / 2
                    box_xy = box_xy / tf.constant([[config.img_w, config.img_h]], dtype=tf.float32)
                    anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                    print(anchor_idx.get_shape().as_list())
                    # todo: anchor_idx = y_true[bs_idx][box_idx][5]
                    grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)
                return 1
            def not_transform():
                return -1
            d = tf.cond(pred=condition, true_fn=, false_fn=)


            print(box_idx)
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[bs_idx][box_idx][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[bs_idx][box_idx][0:4]
                # box_xy ---> center_point
                box_xy = (y_true[bs_idx][box_idx][0:2] + y_true[bs_idx][box_idx][2:4]) / 2
                box_xy = box_xy / tf.constant([[config.img_w, config.img_h]], dtype=tf.float32)
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                print(anchor_idx.get_shape().as_list())
                # todo: anchor_idx = y_true[bs_idx][box_idx][5]
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)
    
    '''

    #             # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
    #             indexes = indexes.write(
    #                 idx, [bs_idx, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
    #             updates = updates.write(
    #                 idx, [box[0], box[1], box[2], box[3], 1, y_true[bs_idx][box_idx][4]])
    #             idx += 1
    #
    # print(tf.shape(indexes))
    # print(updates.get_shape().as_list())

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    # return tf.tensor_scatter_nd_update(
    #     y_true_out, indexes.stack(), updates.stack())


if __name__ == '__main__':

    anchors = config.anchors

    DataSet = Read_Tfrecord(tfrecord_dir='/data/data/weche_tfrecords', anchors=anchors)

    # config.batch_size = 2
    imgs, labels = DataSet.make_batch(batch_size=2)

    # labels = transform_targets(y_train=labels, anchors=anchors)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        _imgs, _labels = sess.run([imgs, labels])
        print('_imgs: ', _imgs.shape)
        # print('_labels_0: ', _labels[0].shape)
        # print('_labels_1: ', _labels[1].shape)
        # print('_labels_2: ', _labels[2].shape)

        # print(_labels[0][1, :, :])
        img_0 = _imgs[0]
        # img_0=np.asarray(img_0, dtype=np.uint8)
        img_0 = cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR)
        cv2.imshow('img_0', img_0)
        cv2.waitKey(0)

        coord.request_stop()
        coord.join(threads=threads)
