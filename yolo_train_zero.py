import tensorflow as tf
import os
import cv2
import numpy as np

import config
from data_process import load_dataset
from model_design import model_design_nn, calc_loss



def main():


    anchors = [[12, 10], [18, 15], [22, 22], [29, 20], [29, 29], [37, 24], [36, 34], [40, 43], [44, 55]]

    DataSet = load_dataset.Read_Tfrecord(tfrecord_dir='/data/data/weche_tfrecords', anchors=anchors)

    config.batch_size = 2
    imgs, y_true = DataSet.make_batch(batch_size=config.batch_size)

    fm1, fm2, fm3 = model_design_nn.yolo3_net(inputs=imgs)
    # anchors = [[20, 30], [40, 50], [60, 70]]
    y_pred = [fm3, fm2, fm1]
    loss = calc_loss.build_loss(y_pred=y_pred, y_true=y_true)

    tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
    tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
    tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
    tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
    tf.summary.scalar('train_batch_statistics/loss_class', loss[4])

    # summary_op
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train_summary')

    # train_op
    global_step = tf.train.get_or_create_global_step()
    # add update_op for slim.bn
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
            loss=loss[0],
            global_step=global_step
        )

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)
    model_save_dir = './run_output'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    ckpt_path = tf.train.latest_checkpoint(model_save_dir)
    print('latest_checkpoint_path: ', ckpt_path)
    if ckpt_path is not None:
        saver.restore(sess, ckpt_path)
        prev_step = int(ckpt_path.split('-')[-1])
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        prev_step = -1

    with sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(100):
            _loss, _, _summary_op = sess.run([loss[0], train_op, summary_op])
            print('step: ', prev_step + 1 + i, 'loss: ', _loss)

            # train_writer.add_summary(_summary_op, prev_step + 1 + i)
            # train_writer.flush()

            if i % 10 == 0:
                saver.save(sess=sess,
                           save_path=os.path.join(model_save_dir, 'model.ckpt'),
                           global_step=global_step)

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()




if __name__=='__main__':

    main()

