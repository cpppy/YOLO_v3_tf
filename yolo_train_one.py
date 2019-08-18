import tensorflow as tf
import os
import cv2
import numpy as np

import config
from data_process import load_dataset
from model_design import model_design_nn, calc_loss

import define_input_fn
import hparams

import logging

logging.getLogger().setLevel(logging.INFO)


def train_ver1():

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


def yolo3_model(mode, feature, label, batch_size):
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    fm1, fm2, fm3 = model_design_nn.yolo3_net(inputs=feature)
    y_pred = [fm3, fm2, fm1]

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # calc ctc loss
        loss = calc_loss.build_loss(y_pred=y_pred, y_true=label)

        # # calc l2 loss
        # l2_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
        # for scope_name in ['CNN_Net', 'RNN_Net' 'FC_Net']:
        #     net_tv = tf.trainable_variables(scope=scope_name)
        #     r_lambda = 0.001
        #     regularization_cost = r_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in net_tv])
        #     loss += regularization_cost
        #     l2_loss += regularization_cost

        return y_pred, loss  # , l2_loss, net_out, tensor_dict, seq_len

    else:
        # inference
        loss = None
        # l2_loss = None
        return y_pred, loss  # , l2_loss, net_out, tensor_dict, seq_len


def my_model_fn(features, labels, mode, params):
    y_pred, loss = yolo3_model(mode=mode,
                               feature=features,
                               label=labels,
                               batch_size=params.batch_size)

    if (mode == tf.estimator.ModeKeys.TRAIN):
        global_step = tf.train.get_global_step()
        starter_learning_rate = params.learning_rate

        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step,
                                                   params.decay_steps,
                                                   params.decay_rate)

        total_loss = loss[0]
        tf.summary.scalar('calc_lr', learning_rate)
        # tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('loss', total_loss)

        # TODO: optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                global_step=global_step)

        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # train_op = tf.contrib.slim.learning.create_train_op(loss,
        #                                                     optimizer,
        #                                                     summarize_gradients=True)

        summary_hook = tf.train.SummarySaverHook(save_steps=params.summary_freq,
                                                 output_dir=config.train_summary_dir,
                                                 summary_op=tf.summary.merge_all())

        # log define
        tensors_to_log = {'global_step': global_step,
                          'lr': learning_rate,
                          'loss': total_loss,
                          # 'l2_loss': l2_loss,
                          # 'train_acc': acc
                          }

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=params.log_freq)
        train_hooks = [logging_hook, summary_hook]

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=total_loss,
                                          train_op=train_op,
                                          training_hooks=train_hooks
                                          )

    elif mode == tf.estimator.ModeKeys.EVAL:

        tensors_to_log = {'eval_loss': loss[0]}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                                  every_n_iter=params.log_freq
                                                  )
        eval_hooks = [logging_hook]

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss[0],
                                          evaluation_hooks=eval_hooks
                                          )


    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred
                                          )

    else:
        print('ERROR: estimator_mode is invalid, task over')
        raise Exception


def main():
    # load params
    _hparams = hparams.HParams()
    # _hparams.reload_params(config.config_fpath)

    tf.keras.optimizers.Adadelta()
    tf.train.AdadeltaOptimizer()

    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,
        gpu_options=tf.GPUOptions(force_gpu_compatible=False))
    # sess_config.gpu_options.per_process_gpu_memory_fraction=0.4
    run_config = tf.estimator.RunConfig(session_config=sess_config,
                                        save_checkpoints_steps=_hparams.save_checkpoints_steps,
                                        keep_checkpoint_max=_hparams.keep_checkpoint_max,
                                        model_dir=config.model_save_dir)

    estimator = tf.estimator.Estimator(
        model_fn=my_model_fn,
        config=run_config,
        params=_hparams, )

    # train_setting
    BATCH_SIZE = _hparams.batch_size * _hparams.gpu_nums  # 16
    # EPOCHS = 5
    train_steps = _hparams.train_steps  # 2000
    eval_steps = _hparams.eval_steps

    tfrecord_dir = config.tfrecord_dir

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: define_input_fn.my_input_fn(data_dir=tfrecord_dir,
                                                                                     batch_size=BATCH_SIZE),
                                        max_steps=train_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: define_input_fn.my_input_fn(data_dir=tfrecord_dir,
                                                                                   batch_size=BATCH_SIZE),
                                      steps=eval_steps,
                                      start_delay_secs=1)

    tf.estimator.train_and_evaluate(estimator,
                                    train_spec,
                                    eval_spec)


if __name__ == '__main__':
    # gen_tf_data.main()
    main()
