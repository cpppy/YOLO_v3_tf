import multiprocessing
import os
import shutil
import json
import math

import sys
sys.path.append('../')

import config
from batch_convert import gen_tfrecord_unit


def init_directory(save_dir):
    # init directory of train_data
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


def worker(sign, thread_id, dataset_dir, tfrecord_dir, sample_idx_range, subset):
    # print("thread_%d" % thread_id, 'generating task, start ... ')
    write_obj = gen_tfrecord_unit.Write_Tfrecord(dataset_dir=dataset_dir,
                                                 tfrecord_dir=tfrecord_dir,
                                                 thread_id=thread_id,
                                                 sample_idx_range=sample_idx_range,
                                                 subset=subset
                                                 )

    write_obj.convert_to_tfrecord()


def setup_gen_task(pid_num, train_data_dir, tfrecord_dir, subset):
    print('################ setup multi_generating_task ###############')
    print('n_pid: ', pid_num)

    img_dir = os.path.join(train_data_dir, 'images')
    img_fn_list = os.listdir(img_dir)
    print('n_samples: ', len(img_fn_list))
    n_sample = len(img_fn_list)
    n_bs_per_process = math.ceil(n_sample / pid_num)
    print('n_bs_per_process: ', n_bs_per_process)

    pid_list = []
    for i in range(pid_num):
        sample_idx_range = [i * n_bs_per_process, (i + 1) * n_bs_per_process]
        print('pid_%d: ', i+1, '%d ------> %d'%tuple(sample_idx_range))
        curr_pid = multiprocessing.Process(target=worker,
                                           args=('process',
                                                 i + 1,
                                                 train_data_dir,
                                                 tfrecord_dir,
                                                 sample_idx_range,
                                                 subset)
                                           )
        pid_list.append(curr_pid)
    for pid in pid_list:
        pid.start()
    for pid in pid_list:
        pid.join()
    # wait all task over
    print('all task(%s) over' % subset)


if __name__ == '__main__':

    init_directory(save_dir=config.tfrecord_dir)
    setup_gen_task(
        pid_num=config.pid_num,
        train_data_dir=config.train_data_dir,
        tfrecord_dir=config.tfrecord_dir,
        subset='train'
    )


