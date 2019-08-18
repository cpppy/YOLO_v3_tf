import tensorflow as tf

import sys
sys.path.append('../')

import config
from model_design import model_design_nn



def broadcast_iou(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh):
    '''
    :param pred_box_xy: [N, 13, 13, 3, 2]
    :param pred_box_wh: [N, 13, 13, 3, 2]
    :param true_box_xy: [M, 2]
    :param true_box_wh: [M, 2]
    :return iou: [N, 13, 13, 3, M]
    '''

    # shape: [N, 13, 13, 3, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)
    # shape: [M, 2] --> [1, M, 2]
    true_box_xy = tf.expand_dims(true_box_xy, 0)
    true_box_wh = tf.expand_dims(true_box_wh, 0)

    # [N, 13, 13, 3, 1, 2] & [1, M, 2] ==> [N, 13, 13, 3, M, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [N, 13, 13, 3, M]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # shape: [N, 13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    # shape: [1, M]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

    # calc iou
    # [N, 13, 13, 3, 1] & [1, M] ---> [N, 13, 13, 3, M]
    union_area = pred_box_area + true_box_area
    union_area = union_area - intersect_area
    # iou: [N, 13, 13, 3, M]
    iou = intersect_area / (union_area + 1e-10)

    return iou


def calc_loss_for_each_feature_map(feature_map_i, y_true_i, anchor_i):
    '''
    :param feature_map_i: [N, 13, 13, 3, 5 + cls_num]
    :param y_true_i: [N, 13, 13, 3, 5 + cls_num]
    :param anchors: [3, 2]
    :return:
    '''
    grid_size = tf.shape(feature_map_i)[1:3]
    ratio = tf.cast([config.img_h / grid_size[0], config.img_w / grid_size[1]], tf.float32)
    N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)  # batch_size

    x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = model_design_nn.gen_bboxes_and_rescale2origin(
        feature_map=feature_map_i,
        anchors=anchor_i
    )

    ###########
    # get mask
    ###########
    # shape: take 416x416 input image and 13*13 feature_map for example:
    # [N, 13, 13, 3, 1]
    object_mask = y_true_i[..., 4:5]   # conf

    # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [M, 4]
    # V: num of true gt box
    print('y_true_i: ', y_true_i.get_shape().as_list())
    valid_true_boxes = tf.boolean_mask(y_true_i[..., 0:4], tf.cast(object_mask[..., 0], dtype=tf.bool))
    # print('valid_true_boxes: ', valid_true_boxes.get_shape().as_list())

    # shape: [M, 2]
    valid_true_box_xy = valid_true_boxes[:, 0:2]
    valid_true_box_wh = valid_true_boxes[:, 2:4]
    # shape: [N, 13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # calc iou
    # shape: [N, 13, 13, 3, M]
    iou = broadcast_iou(pred_box_xy=pred_box_xy,
                        pred_box_wh=pred_box_wh,
                        true_box_xy=valid_true_box_xy,
                        true_box_wh=valid_true_box_wh)

    # shape: [N, 13, 13, 3]
    best_iou = tf.reduce_max(iou, axis=-1)

    # get_ignore_mask
    ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
    # shape: [N, 13, 13, 3, 1]
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    # get xy coordinates in one cell from the feature_map
    # numerical range: 0 ~ 1
    # shape: [N, 13, 13, 3, 2]
    true_xy = y_true_i[..., 0:2] / ratio[::-1] - x_y_offset  # gt_box_center_point (x,y) in current cell, cell_num:13*13
    pred_xy = pred_box_xy / ratio[::-1] - x_y_offset   # pred_box_center_point (x, y)

    # get_tw_th
    # numerical range: 0 ~ 1
    # shape: [N, 13, 13, 3, 2]/[3, 2] ---> [N, 13, 13, 3, 2]
    true_tw_th = y_true_i[..., 2:4] / anchor_i
    pred_tw_th = pred_box_wh / anchor_i
    # for numerical stability
    true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                          x=tf.ones_like(true_tw_th), y=true_tw_th)
    pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                          x=tf.ones_like(pred_tw_th), y=pred_tw_th)
    true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-5, 1e5))  # todo: (1e-3, 1e3)
    pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-5, 1e5))

    # box size punishment:
    # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
    # shape: [N, 13, 13, 3, 1]
    box_loss_scale = 2. - (y_true_i[..., 2:3] / tf.cast(config.img_w, tf.float32)) * (
            y_true_i[..., 3:4] / tf.cast(config.img_h, tf.float32))

    ############
    # loss_part
    ############
    # shape: [N, 13, 13, 3, 1]
    xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
    wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

    # shape: [N, 13, 13, 3, 1]
    conf_pos_mask = object_mask # object_mask * valid_mask
    conf_neg_mask = (1 - object_mask) * ignore_mask
    conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
    conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
    conf_loss = tf.reduce_sum(conf_loss_pos + conf_loss_neg) / N

    # shape: [N, 13, 13, 3, 1]
    class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_i[..., 5:],
                                                                       logits=pred_prob_logits)
    class_loss = tf.reduce_sum(class_loss) / N

    return xy_loss, wh_loss, conf_loss, class_loss



def build_loss(y_pred, y_true):
    '''
    param:
        y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
        y_true: input y_true by the tf.data pipeline
    '''
    loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
    anchors = config.anchors
    anchor_group = [anchors[0:3], anchors[3:6], anchors[6:9]]

    # calc loss in 3 scales
    for i in range(len(y_pred)):
        result = calc_loss_for_each_feature_map(feature_map_i=y_pred[i],
                                                y_true_i=y_true[i],
                                                anchor_i=anchor_group[i])
        loss_xy += result[0]
        loss_wh += result[1]
        loss_conf += result[2]
        loss_class += result[3]
    total_loss = loss_xy + loss_wh + loss_conf + loss_class
    return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]


if __name__=='__main__':

    y_pred = [tf.placeholder(dtype=tf.float32, shape=[None, 52, 52, 3, 134]),
              tf.placeholder(dtype=tf.float32, shape=[None, 26, 26, 3, 134]),
              tf.placeholder(dtype=tf.float32, shape=[None, 13, 13, 3, 134])]

    y_true = [tf.placeholder(dtype=tf.float32, shape=[None, 52, 52, 3, 134]),
              tf.placeholder(dtype=tf.float32, shape=[None, 26, 26, 3, 134]),
              tf.placeholder(dtype=tf.float32, shape=[None, 13, 13, 3, 134])]

    loss = build_loss(y_pred=y_pred, y_true=y_true)

    total_loss, loss_xy, loss_wh, loss_conf, loss_class = loss

    print(total_loss.shape)














