# coding=utf-8
import tensorflow as tf

import sys
sys.path.append('../')

import config
from model_design import darknet_base
import cls_dict
from model_design import calc_loss

n_feature_map_channel = 3 * (5 + len(cls_dict.label_num_dict))


def yolo3_net(inputs):
    with tf.variable_scope('yolo3_net'):
        # ---------------- darknet ------------------
        route_1, route_2, route_3 = darknet_base.darknet_53(inputs=inputs)
        print('darknet output: ')
        print(route_1.get_shape().as_list())
        print(route_2.get_shape().as_list())
        print(route_3.get_shape().as_list())

        # ---------------- branch 1 ------------------
        branch_1, block_out_1 = darknet_base.yolo_block(inputs=route_3, filters=512)
        feature_map_1 = tf.keras.layers.Conv2D(filters=n_feature_map_channel,
                                       kernel_size=(1, 1),
                                       strides=(1, 1),
                                       name='%s_%s' % ('branch1', 'conv1_1')
                                       )(block_out_1)

        # feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

        # ------------------- branch 2 --------------------
        conv2 = darknet_base.conv2d(inputs=branch_1,
                                    filters=256,
                                    kernel_size=1,
                                    strides=1,
                                    name='%s_%s' % ('branch2', 'conv2_1')
                                    )

        upsample_2 = darknet_base.upsample_layer(conv2, route_2.get_shape().as_list())

        concat_2 = tf.concat([upsample_2, route_2], axis=3)

        branch_2, block_out_2 = darknet_base.yolo_block(inputs=concat_2, filters=256)
        feature_map_2 = tf.keras.layers.Conv2D(filters=n_feature_map_channel,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               name='branch2_conv2_2'
                                               )(block_out_2)

        # feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

        # --------------------- branch 3 ---------------------
        conv3 = darknet_base.conv2d(inputs=branch_2,
                                    filters=128,
                                    kernel_size=1,
                                    strides=1,
                                    name='brank3_conv3_1'
                                    )
        upsample_3 = darknet_base.upsample_layer(inputs=conv3, out_shape=route_1.get_shape().as_list())

        concat_3 = tf.concat([upsample_3, route_1], axis=3)

        _, block_out_3 = darknet_base.yolo_block(inputs=concat_3, filters=128)

        feature_map_3 = tf.layers.Conv2D(filters=n_feature_map_channel,
                                         kernel_size=(1, 1),
                                         strides=(1, 1),
                                         name='branch3_conv3_2'
                                         )(block_out_3)

        # feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

        print('feature_map_1: ', feature_map_1.get_shape().as_list())
        print('feature_map_2: ', feature_map_2.get_shape().as_list())
        print('feature_map_3: ', feature_map_3.get_shape().as_list())
        return feature_map_1, feature_map_2, feature_map_3


def gen_bboxes_and_rescale2origin(feature_map, anchors):
    grid_size = feature_map.shape.as_list()[1:3]  # [13, 13]
    ratio = tf.cast([config.img_h / grid_size[0], config.img_w / grid_size[1]], tf.float32)  # input/grid[
    # rescale the anchors to the feature_map
    grid_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + len(cls_dict.label_num_dict)])

    # split the feature_map along the last dimension
    # shape info: take 416x416 input image and the 13*13 feature_map for example:
    # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
    # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
    # conf_logits: [N, 13, 13, 3, 1]
    # prob_logits: [N, 13, 13, 3, class_num]
    box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map,
                                                                [2, 2, 1, len(cls_dict.label_num_dict)],
                                                                axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)

    # use some broadcast tricks to get the mesh coordinates
    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    # print('grid_x', grid_x.get_shape().as_list())
    # print('grid_y', grid_y.get_shape().as_list())
    mesh_x, mesh_y = tf.meshgrid(grid_x, grid_y)
    # print('mesh_x', mesh_x.get_shape().as_list())
    # print('mesh_y', mesh_y.get_shape().as_list())
    x_offset = tf.reshape(mesh_x, (-1, 1))
    y_offset = tf.reshape(mesh_y, (-1, 1))
    # print('x_offset: ', x_offset.get_shape().as_list())
    # print('y_offset: ', y_offset.get_shape().as_list())
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    # shape: [13, 13, 1, 2]
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)
    print('x_y_offset: ', x_y_offset.get_shape().as_list())

    # get the absolute box coordinates on the feature_map
    # print('box_centers: ', box_centers.get_shape().as_list())
    box_centers = box_centers + x_y_offset
    # rescale to the original image scale
    box_centers = box_centers * ratio[::-1]

    # avoid getting possible nan value with tf.clip_by_value
    box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * grid_anchors
    # rescale to the original image scale
    box_sizes = box_sizes * ratio[::-1]

    # shape: [N, 13, 13, 3, 4]
    # last dimension: (center_x, center_y, w, h)
    boxes = tf.concat([box_centers, box_sizes], axis=-1)
    print('boxes: ', boxes.get_shape().as_list())

    # shape:
    # x_y_offset: [13, 13, 1, 2]
    # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
    # conf_logits: [N, 13, 13, 3, 1]
    # prob_logits: [N, 13, 13, 3, class_num]
    print('conf_logits: ', conf_logits.get_shape().as_list())
    print('proba_logits: ', prob_logits.get_shape().as_list())
    return x_y_offset, boxes, conf_logits, prob_logits


def main():
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, config.img_h, config.img_w, config.img_ch])
    y_true_l = tf.placeholder(dtype=tf.float32,
                              shape=[None, config.img_h // 8, config.img_w // 8, 3, 5 + len(cls_dict.label_num_dict)])
    y_true_m = tf.placeholder(dtype=tf.float32,
                              shape=[None, config.img_h // 16, config.img_w // 16, 3, 5 + len(cls_dict.label_num_dict)])
    y_true_s = tf.placeholder(dtype=tf.float32,
                              shape=[None, config.img_h // 32, config.img_w // 32, 3, 5 + len(cls_dict.label_num_dict)])
    y_true = [y_true_l, y_true_m, y_true_s]

    fm1, fm2, fm3 = yolo3_net(inputs=inputs)
    # anchors = [[20, 30], [40, 50], [60, 70]]
    y_pred = [fm3, fm2, fm1]
    calc_loss.build_loss(y_pred=y_pred, y_true=y_true)
    # xy_offset, bboxes, conf_logits, prob_logits = gen_bboxes_and_rescale2origin(feature_map=fm1, anchors=anchors)
    # print('xy_offset: ', xy_offset.get_shape().as_list())
    # print('bboxes: ', bboxes.get_shape().as_list())
    # print('conf: ', conf_logits.get_shape().as_list())


if __name__ == '__main__':
    main()

    # a = tf.constant(value=[[1, 2]], dtype=tf.float32)
    # b = tf.constant(value=[[10, 20], [20, 30], [30, 40]], dtype=tf.float32)
    # c = tf.add(a, b)
    # sess = tf.Session()
    # with sess.as_default():
    #     sess.run(tf.initialize_all_variables())
    #     _c = sess.run(c)
    #     print(_c)
    #
    # sess.close()
