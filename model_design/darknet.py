# coding: utf-8

import numpy as np
import tensorflow as tf



def conv2d(inputs, filters, kernel_size, strides, name=''):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)

    inputs = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=('same' if strides == 1 else 'valid'),
                                    name=name
                                    )(inputs)
    return inputs


def res_block(inputs, name):
    n_channel = inputs.get_shape().as_list()[-1]
    res_block_shortcut = inputs
    res_block_conv1 = conv2d(inputs=inputs,
                             filters=int(n_channel / 2),
                             kernel_size=1,
                             strides=1,
                             name='%s_conv1' % name)
    res_block_conv2 = conv2d(inputs=res_block_conv1,
                             filters=n_channel,
                             kernel_size=1,
                             strides=1,
                             name='%s_conv2' % name)
    res_block_out = tf.add(res_block_conv2, res_block_shortcut)
    return res_block_out


def darknet_53(inputs):
    # ------------------- part1 ------------------
    conv1_1 = conv2d(inputs=inputs,
                     filters=32,
                     kernel_size=3,
                     strides=1,
                     name='conv1_1')

    conv1_2 = conv2d(inputs=conv1_1,
                     filters=64,
                     kernel_size=3,
                     strides=2,
                     name='conv1_2')

    res_block1_1 = res_block(inputs=conv1_2,
                             name='res_block1_1')


    # ----------------- part2 ------------------
    conv2_1 = conv2d(inputs=res_block1_1,
                     filters=128,
                     kernel_size=3,
                     strides=2,
                     name='conv2_1')

    block = conv2_1
    for block_idx in range(2):
        block = res_block(inputs=block,
                          name='res_block2_%d'%(block_idx+1))

    #----------------- part3 ------------------
    conv3_1 = conv2d(inputs=block,
                     filters=256,
                     kernel_size=3,
                     strides=2,
                     name='conv3_1')

    block = conv3_1
    for block_idx in range(8):
        block = res_block(inputs=block,
                          name='res_block3_%d'%(block_idx+1))

    route1 = block

    # ----------------- part4 -----------------
    conv4_1 = conv2d(inputs=block,
                     filters=512,
                     kernel_size=3,
                     strides=2,
                     name='conv4_1')

    block = conv4_1
    for block_idx in range(8):
        block = res_block(inputs=block,
                          name='res_block4_%d'%(block_idx+1))
    route2 = block

    # ----------------- part5 -----------------
    conv5_1 = conv2d(inputs=block,
                     filters=1024,
                     kernel_size=3,
                     strides=2,
                     name='conv5_1')

    block = conv5_1
    for block_idx in range(4):
        block = res_block(inputs=block,
                          name='res_block5_%d'%(block_idx+1))
    route3 = block

    return route1, route2, route3


def yolo_block(inputs, filters):
    net = conv2d(inputs=inputs, filters=filters * 1,kernel_size=1, strides=1)
    net = conv2d(inputs=net, filters=filters * 2, kernel_size=3, strides=1)
    net = conv2d(inputs=net, filters=filters * 1, kernel_size=1, strides=1)
    net = conv2d(inputs=net, filters=filters * 2, kernel_size=3, strides=1)
    net = conv2d(inputs=net, filters=filters * 1, kernel_size=1, strides=1)
    route = net
    net = conv2d(inputs=net, filters=filters * 2, kernel_size=3, strides=1)
    return route, net

'''

def yolo_block(inputs, filters, block_name):
    conv1 = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   name='%s_%s'%(block_name, 'conv1')
                                   )(inputs)

    conv2 = tf.keras.layers.Conv2D(filters=filters*2,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   name='%s_%s' % (block_name, 'conv3')
                                   )(conv1)

    conv3 = tf.keras.layers.Conv2D(filters=filters*2,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   name='%s_%s' % (block_name, 'conv3')
                                   )(conv1)

'''


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), align_corners=True, name='upsampled')
    # tf.image.resize_bilinear(images=inputs, size=(new_height, new_width), align_corners=True, name='upsampling')
    return inputs


def main():
    inputs = tf.placeholder(shape=[None, 1024, 1024, 3], dtype=tf.float32)
    fm_1, fm_2, fm_3 = darknet_53(inputs=inputs)
    print(fm_1.get_shape().as_list())
    print(fm_2.get_shape().as_list())
    print(fm_3.get_shape().as_list())
    '''
    [None, 128, 128, 256]
    [None, 64, 64, 512]
    [None, 32, 32, 1024]
    '''


if __name__ == '__main__':
    main()
