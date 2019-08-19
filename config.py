import os
import cls_dict


proj_root_path = os.path.abspath(os.path.dirname(__file__))
print('project_root_path: ', proj_root_path)

# YOLO_v3 model parameters

Dataset_name = 'weche_data'

train_data_dir = '/data/data/weche_train_data'
tfrecord_dir = '/data/data/weche_tfrecords'

pid_num = 4

if not os.path.exists(tfrecord_dir):
    os.mkdir(tfrecord_dir)

model_save_dir = './run_output'
train_summary_dir = './train_summary'

dev_mode=True
config_fpath = './train_config.json'

batch_size = 2

data_format = 'NHWC'

tfrecord_img_h = 1280
tfrecord_img_w = 1280

img_h = 1280
img_w = 1280
img_ch = 3


'''
mean_BGR:  70.8539207582204 73.35755032907225 77.56587565044319
std:  46.05407129657253
'''

RGB_MEAN = (77.56, 73.36, 70.85)
STD = 46.05

cls_num = len(cls_dict.label_num_dict)

# anchors = [[12, 10], [18, 15], [22, 22], [29, 20], [29, 29], [37, 24], [36, 34], [40, 43], [44, 55]]
anchors = [[33, 27], [46, 43], [75, 49], [59, 68], [102, 70], [81, 94], [132, 98], [101, 130], [144, 156]]



