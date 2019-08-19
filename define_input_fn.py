import tensorflow as tf
import cv2
import numpy as np

import config
from data_process import load_dataset


def my_input_fn(data_dir,
                subset='train',
                # num_shards=0,
                batch_size=4):
    with tf.device('/cpu:0'):
        # use_distortion = subset == 'train' and use_distortion_for_training
        anchors = config.anchors
        dataset = load_dataset.Read_Tfrecord(tfrecord_dir=data_dir, anchors=anchors, subset=subset)
        input_data, input_labels = dataset.make_batch(batch_size)
        return input_data, input_labels



'''
def input_fn_for_inference(img_cv2):

    train_images = [cv2.resize(img_cv2, (100, 32))]
    train_images = np.asarray(train_images)
    train_labels = np.asarray([''])
    train_imagenames = np.asarray([''])
    print('image_shape: ', train_images[0].shape)
    print("labels_0: ", train_labels[0])
    print("image_name_0: ", train_imagenames[0])
    return train_images, train_labels






def check_my_input_fn():
    # crnn_dataset = CrnnDataSet(data_dir='/data/data/crnn_tfrecords',
    #                            subset='train')
    #
    # imgs, labels = crnn_dataset.make_batch(batch_size=32)
    feature_shards, label_shards = my_input_fn()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        _feature_shards, _label_shards = sess.run([feature_shards, label_shards])
        _imgs, _labels = _feature_shards[0], _label_shards[0]
        scene_labels = load_tf_data.sparse_tensor_to_str(_labels)
        print('scene_labels_len: ', len(scene_labels))
        for index, scene_label in enumerate(scene_labels):
            # img_name = img_names_val[index][0].decode('utf-8')
            # print('{:s} --- {:s}'.format(img_name, scene_label))
            print('scene_label: ', scene_label)

        coord.request_stop()
        coord.join(threads=threads)


def main():
    # train_model = crnn_model.get_Model(training=True)
    # train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
    #                     optimizer=tf.keras.optimizers.Adadelta()
    #                     )
    #
    # train_estimator = tf.keras.estimator.model_to_estimator(train_model, model_dir='/data/output/crnn_estimator_ckp')

    estimator = None

    BATCH_SIZE = 16
    EPOCHS = 5
    STEPS = 2000
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: my_input_fn(batch_size=BATCH_SIZE),
                                        max_steps=STEPS)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: my_input_fn(batch_size=BATCH_SIZE),
                                      steps=1,
                                      start_delay_secs=3
                                      )

    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec,
                                    eval_spec=eval_spec
                                    )


if __name__ == '__main__':
    check_my_input_fn()

    main()
    
'''
