import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm

import sys
sys.path.append('../')

import config


class Write_Tfrecord():

    def __init__(self, dataset_dir, tfrecord_dir, is_train=True):
        self.dataset_dir = dataset_dir
        self.tfrecord_dir = tfrecord_dir
        self.is_train = is_train

        if not os.path.exists(self.tfrecord_dir):
            os.mkdir(self.tfrecord_dir)

    @staticmethod
    def int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def image_and_gtbox_label(self, image_name_with_format, img_dir, gt_dir):
        """Read the annotation of the image

        Args:
            Str image_name_with_format: image name without format
            List shape: [h, w]
        Returns:
            List gtbox_label: annotations of image
                [[x1,y1,x2,y2,x3,y3,x4,y4,cls_name,cls_idx], [...]]
        """
        image_name, _ = os.path.splitext(image_name_with_format)
        # image_data = tf.gfile.FastGFile(os.path.join(img_dir, image_name_with_format), 'rb').read()
        # array_image = scipy.misc.imread(io.BytesIO(image_data))
        # scm.imshow(array_image)

        img_fpath = os.path.join(img_dir, image_name_with_format)
        img_cv2 = cv2.imread(img_fpath, cv2.IMREAD_COLOR)
        h, w = img_cv2.shape[: 2]

        gtbox_labels = []
        txt_path = os.path.join(gt_dir, "{}.txt".format(image_name))
        with open(txt_path, 'r', encoding='utf-8') as annot:
            for line in annot:
                line = line.strip().split(',')
                box, label = list(map(float, line[: 8])), int(line[-1])
                # assert np.array(box).any() >= 0
                box = np.array(box) / ([w, h] * 4)
                box = np.concatenate([box, [label]])
                gtbox_labels.append(box)

        # process img_data
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        rescaled_img_cv2 = cv2.resize(img_cv2, (config.tfrecord_img_w, config.tfrecord_img_h))
        img_data = np.asarray(rescaled_img_cv2, np.int)
        img_data = bytes(list(np.reshape(img_data, [config.img_h*config.img_w*config.img_ch])))
        return image_name, img_data, gtbox_labels

    @staticmethod
    def get_list(obj, idx):
        obj = np.asarray(obj)
        if len(obj) > 0:
            # print (list(obj[:, idx]))
            return list(obj[:, idx])
        return []

    def convert_to_tfexample(self, image_data, image_name, image_format, gtbox_label):
        if len(gtbox_label) == 0:
            print(image_name, "has no gtbox_label")
        example = tf.train.Example(features=tf.train.Features(feature={
            "image/filename": self.bytes_feature(image_name),
            "image/encode": self.bytes_feature(image_data),
            "image/format": self.bytes_feature(image_format),
            "label/x1": self.float_feature(self.get_list(gtbox_label, 0)),
            "label/y1": self.float_feature(self.get_list(gtbox_label, 1)),
            "label/x2": self.float_feature(self.get_list(gtbox_label, 2)),
            "label/y2": self.float_feature(self.get_list(gtbox_label, 3)),
            "label/x3": self.float_feature(self.get_list(gtbox_label, 4)),
            "label/y3": self.float_feature(self.get_list(gtbox_label, 5)),
            "label/x4": self.float_feature(self.get_list(gtbox_label, 6)),
            "label/y4": self.float_feature(self.get_list(gtbox_label, 7)),
            "label/cls": self.int64_feature(list(map(int, self.get_list(gtbox_label, 8)))),
        }))
        return example

    def convert_to_tfrecord(self):
        img_dir = os.path.join(self.dataset_dir, 'images')
        assert tf.gfile.Exists(img_dir), "Image folder {} are not exits;Or your images' folder name is not JPEG".format(
            img_dir)
        gt_dir = os.path.join(self.dataset_dir, 'labels')
        assert tf.gfile.Exists(
            gt_dir), "Annotations folder {} are not exits;Or your annotations' folder name is not ANNOT".format(gt_dir)

        images = tf.gfile.ListDirectory(img_dir)
        # tfrecord_path = tools.make_folder(self.tfrecord_dir)
        if self.is_train:
            tfrecord_path = os.path.join(os.path.abspath(self.tfrecord_dir),
                                         "{}_{}.tfrecord".format(config.Dataset_name, 'train'))
        else:
            tfrecord_path = os.path.join(os.path.abspath(self.tfrecord_dir),
                                         "{}_{}.tfrecord".format(config.Dataset_name, 'test'))
        print('{}\nTfrecord Saving Path:\n\t{}\n{}'.format("@@" * 30, tfrecord_path, "@@" * 30))

        # with tf.io.TFRecordWriter(tfrecord_path) as tfrecord_writer: # NOTE tf1.12+
        with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
            pbar = tqdm(images, total=len(images), ncols=80)
            for image_name_with_format in pbar:
                image_name, image_data, gtbox_label = self.image_and_gtbox_label(image_name_with_format, img_dir,
                                                                                 gt_dir)
                pbar.set_description(image_name)
                # cv2.imshow('1', array_image)
                # cv2.waitKey(100000)
                # print (np.shape(gtbox_label))
                example = self.convert_to_tfexample(image_name=image_name.encode(),
                                                    image_data=image_data,
                                                    image_format='jpg'.encode(),
                                                    gtbox_label=gtbox_label,
                                                    )
                tfrecord_writer.write(example.SerializeToString())


def do_generate():
    write_obj = Write_Tfrecord(dataset_dir=config.train_data_dir, tfrecord_dir=config.tfrecord_dir)

    write_obj.convert_to_tfrecord()


if __name__=='__main__':

    write_obj = Write_Tfrecord(dataset_dir='/data/data/weche_train_data',
                               tfrecord_dir='/data/data/weche_tfrecords')

    write_obj.convert_to_tfrecord()




