import tensorflow as tf
import numpy as np


import sys
sys.path.append('../')
# from data_prepare import bbox_convert
# from data_prepare import tools
from data_process.preprocess import encode_single_image
import config


def preprocess_image(image,
                     labels=None,
                     bboxes=None,
                     xs=None, ys=None,
                     out_shape=None,
                     data_format='NHWC',
                     is_training=False):

    return preprocess_for_train_no_aug(image, labels, bboxes, xs, ys,
                                       out_shape=out_shape,
                                       data_format=data_format)

def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height, width, channels = image.get_shape().as_list()
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image



def tf_image_whitened(image, means=config.RGB_MEAN):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    image = image / tf.constant(config.STD, dtype=image.dtype)
    return image


def tf_imagenet_preprocess(image, means=config.RGB_MEAN):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    #print('image_dtype: ', image.dtype)  # float32
    image = image - mean
    image = image / tf.constant(config.STD, dtype=image.dtype)
    return image


#
#
# def encode_single_image_old(normed_xs, normed_ys, labels):
#     """
#     Args:
#         xs, ys: both in shape of (N, 4),
#             and N is the number of bboxes,
#             their values are normalized to [0,1]
#         labels: shape = (N,)
#              -1: ignored
#               1: text
#     Return:
#         cls_label
#         cls_weight
#         link_label
#         link_weight
#         rects_with_labels:(N, 6) each one is [x_c, y_c, w, h, theta, label]
#     """
#     # validate the args
#     assert np.ndim(normed_xs) == 2
#     assert np.shape(normed_xs)[-1] == 4
#     assert np.shape(normed_xs) == np.shape(normed_ys)
#     assert len(normed_xs) == len(labels)
#
#     xs = normed_xs * W
#     ys = normed_ys * H
#
#     cls_label = np.zeros([H, W, NUM_CLASSES], dtype=np.int32)
#     cls_weight = np.zeros([H, W, NUM_CLASSES], dtype=np.float32)
#     link_label = np.zeros((H, W, NUM_NEIGHBOURS), dtype=np.int32)
#     link_weight = np.ones((H, W, NUM_NEIGHBOURS), dtype=np.float32)
#
#     bbox_masks = []
#     border_masks = []
#     mask = np.zeros([H, W], dtype=np.int32)
#     rects_with_labels = []
#     for cls, num in label_num_map.items():
#         if num <= 0:  # background
#             continue
#         for bbox_idx, (bbox_xs, bbox_ys) in enumerate(list(zip(xs, ys))):
#             # 1: preprocessing coordinates & filter background and ignore
#             # if np.max(bbox_xs) < 0 or np.min(bbox_xs) > W:
#             #     continue
#             # elif np.max(bbox_ys) < 0 or np.min(bbox_ys) > H:
#             #     continue
#             # else:
#             bbox_xs, bbox_ys = np.clip(bbox_xs, 0, W), np.clip(bbox_ys, 0, H)
#
#             # 2: [x1,y1,x2,y2,x3,y3,x4,y4,label] => [x_c,y_c,w,h, theta, label]
#             # np.stack((bbox_xs, bbox_ys), axis=1) ---> [[x0,y0], [x1,y1], [x2,y2]...]
#             eight_cords_rect = np.hstack((np.stack((bbox_xs, bbox_ys), axis=1).flatten(), labels[bbox_idx])).reshape([-1, 9])
#             # eight_cords_rect: [N, [x1,y1,x2,y2,x3,y3,x4,y4,label=1]]
#             angle_cords = bbox_convert.back_forward_convert(eight_cords_rect, True).flatten()
#             # angle_cords: [N, [x,y,w,h,theta,label=1]]
#             rects_with_labels.append(angle_cords)
#             # print ("{}\nTwo Coordinate Type:\n\t{}\n\t{}\n".format("==="*20, eight_cords_rect, angle_cords, "==="*20))
#
#             if labels[bbox_idx] == num:
#                 label = 1
#                 level = 0
#
#                 # 3: generate bbox mask and border mask
#                 bbox_mask_each_bbox = mask.copy()
#                 bbox_contours = tools.points_to_contours(list(zip(bbox_xs, bbox_ys)))
#                 # bbox_contours: [N, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]]
#                 tools.draw_contours(bbox_mask_each_bbox, bbox_contours, idx=-1, color=1, border_width=-1)
#
#                 MINIMUM_MASK_REMAINED=20
#                 if np.sum(bbox_mask_each_bbox) <= MINIMUM_MASK_REMAINED:
#                     continue
#                 bbox_masks.append(bbox_mask_each_bbox)
#                 bbox_border_mask_each_bbox = mask.copy()
#                 tools.draw_contours(bbox_border_mask_each_bbox, bbox_contours, -1, color=1,
#                                   border_width=bbox_border_width)
#                 border_masks.append(bbox_border_mask_each_bbox)
#
#                 # 4: encode cls_label and cls_weight
#                 cls_label[:, :, level] = np.clip(
#                     np.where(bbox_mask_each_bbox, cls_label[:, :, level] + label, cls_label[:, :, level]),
#                     a_min=0,
#                     a_max=label)
#                 '''
#                 if cls_weight_with_border_balanced:
#                     # ---------------------avoid overlap area have too large weight------------------------------
#                     # TODO address overlap areas
#                     before_max = np.max(cls_weight[:, :, level])
#                     after_max = 1 / np.sum(bbox_mask_each_bbox) if np.sum(bbox_mask_each_bbox) > 0 else 1
#                     cls_weight[:, :, level] += bbox_mask_each_bbox * after_max
#                     np.clip(cls_weight[:, :, level], a_min=0, a_max=np.max([before_max, after_max]))
#                     # ---------------------set larger weight for border area ------------------------------------
#                     cls_weight[:, :, level] += bbox_border_mask_each_bbox * CLS_BORDER_WEITGHT_LAMBDA
#                 '''
#                 # 5: encode link_label and link_weight
#                 if USE_LINK:
#                     bbox_cls_cords = np.where(bbox_mask_each_bbox)
#                     link_label[bbox_cls_cords] = 1
#                     bbox_border_cords = np.where(bbox_border_mask_each_bbox)
#                     border_points = list(zip(*bbox_border_cords))
#                     for y, x in border_points:
#                         neighbours = tools.get_neighbours(x, y)
#                         for n_idx, (nx, ny) in enumerate(neighbours):
#                             if not tools.is_valid_cord(nx, ny, W, H) or not bbox_mask_each_bbox[nx, ny]:
#                                 link_label[y, x, n_idx] = 0
#
#     # num_positive_bboxes_each_cls = Counter(labels)
#     # num_positive_pixels_each_cls = np.sum(cls_label, axis=2)
#     num_total_positive_pixels = np.sum(cls_label)
#     # overlap area will greater then 1
#     '''
#     if cfgs.cls_weight_with_border_balanced:
#         cls_weight = num_total_positive_pixels * cls_weight
#         cls_weight = np.clip(cls_weight, a_min=0, a_max=WEIGHT_THRESHHOLD)
#     else:
#     '''
#     cls_weight += cls_label
#     cls_weight = np.cast["float32"](cls_weight)
#     if USE_LINK:
#         link_weight *= cls_weight
#
#     return cls_label, cls_weight, link_label, link_weight, rects_with_labels


def tf_encode_single_image(xs, ys, labels):
    cls_label, cls_weight, link_label, link_weight, rects_with_labels = tf.py_func(encode_single_image,
                                                                                   [xs, ys, labels],
                                                                                   [tf.int32, tf.float32, tf.int32,
                                                                                    tf.float32, tf.float32])
    H, W, NUM_CLASSES, NUM_NEIGHBOURS = config.img_h, config.img_w, 1, 8
    cls_label.set_shape([H, W, NUM_CLASSES])
    cls_weight.set_shape([H, W, NUM_CLASSES])
    link_label.set_shape([H, W, NUM_NEIGHBOURS])
    link_weight.set_shape([H, W, NUM_NEIGHBOURS])
    rects_with_labels = tf.reshape(rects_with_labels, [-1, 6])
    return cls_label, cls_weight, link_label, link_weight, rects_with_labels


def preprocess_for_train_no_aug(image, labels, bboxes, xs, ys,
                                out_shape, data_format='NHWC',
                                scope='ssd_preprocessing_train'):


    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        # Convert to float scaled [0, 1].
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Resize image to output size.
        dst_image = resize_image(image, out_shape,
                                 method=tf.image.ResizeMethod.BILINEAR,
                                 align_corners=False)

        # =======================NOTE method 7:whiten
        # Rescale to VGG input scale.
        # image = dst_image * 255.
        # image = tf_image_whitened(image, config.RGB_MEAN)
        image = tf_imagenet_preprocess(image, config.RGB_MEAN)
        return image, labels, bboxes, xs, ys
