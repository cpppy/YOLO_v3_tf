import tensorflow as tf
import os
import cv2
import numpy as np

import config
from model_design import model_design_nn, calc_loss, object_parser
from utils import nms_utils

import define_input_fn
import hparams

import logging

logging.getLogger().setLevel(logging.INFO)


def yolo3_model(mode, feature, label, batch_size):
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    fm1, fm2, fm3 = model_design_nn.yolo3_net(inputs=feature)
    y_pred = [fm3, fm2, fm1]

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # calc ctc loss
        loss = calc_loss.build_loss(y_pred=y_pred, y_true=label)

        # calc l2 loss
        l2_loss = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)
        for scope_name in ['CNN_Net', 'RNN_Net' 'FC_Net']:
            net_tv = tf.trainable_variables(scope=scope_name)
            r_lambda = 0.001
            regularization_cost = r_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in net_tv])
            loss += regularization_cost
            l2_loss += regularization_cost

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

        pred_boxes, pred_confs, pred_probs = object_parser.predict(feature_maps=y_pred)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = nms_utils.gpu_nms(pred_boxes,
                                                  pred_scores,
                                                  config.cls_num,
                                                  max_boxes=3,
                                                  score_thresh=0.01,
                                                  iou_thresh=0.5)

        predictions = {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions
                                          )

    else:
        print('ERROR: estimator_mode is invalid, task over !')
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

    test_img_dir = '/data/data/weche_train_data/images'
    img_fn_list = os.listdir(test_img_dir)[0:1]
    for img_fn in img_fn_list:
        print('img_fn: ', img_fn)
        img_fpath = os.path.join(test_img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = cv2.resize(img_cv2, (config.img_w, config.img_h))
        img_data = np.asarray(img_cv2, dtype=np.float)
        img_data = np.expand_dims(img_data, axis=0)
        test_images = np.asarray(img_data, dtype=np.float32)
        test_labels = np.asarray([''] * img_data.shape[0])

        test_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_images,
                                                           y=test_labels,
                                                           batch_size=1,
                                                           num_epochs=1,
                                                           shuffle=False,
                                                           queue_capacity=1,
                                                           num_threads=1)
        predictions = estimator.predict(input_fn=test_input_fn,
                                        yield_single_examples=False)
        _predictions = next(predictions)
        pred_boxes = _predictions['boxes']
        print('boxes: ', pred_boxes)
        print('labels: ', _predictions['labels'])
        print('scores: ', _predictions['scores'])



    # tfrecord_dir = config.tfrecord_dir
    #
    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: define_input_fn.my_input_fn(data_dir=tfrecord_dir,
    #                                                                                  batch_size=BATCH_SIZE),
    #                                     max_steps=train_steps)
    #
    # eval_spec = tf.estimator.EvalSpec(input_fn=lambda: define_input_fn.my_input_fn(data_dir=tfrecord_dir,
    #                                                                                batch_size=BATCH_SIZE),
    #                                   steps=eval_steps,
    #                                   start_delay_secs=1)
    #
    # tf.estimator.train_and_evaluate(estimator,
    #                                 train_spec,
    #                                 eval_spec,
    #                                 )


if __name__ == '__main__':
    # gen_tf_data.main()
    main()
