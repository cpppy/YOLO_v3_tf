import numpy as np
import cv2
import os


def res_area(img_cv2):
    lower_blue = np.array([156, 43, 46])
    upper_blue = np.array([180, 255, 255])
    lower_red = np.array([11, 43, 46])
    upper_red = np.array([25, 255, 255])
    lower_red3 = np.array([0, 43, 46])
    upper_red3 = np.array([10, 255, 255])
    # image = cv2.imread(pic)
    image = img_cv2.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
    mask = mask1 + mask2 + mask3
    res = cv2.bitwise_and(image, image, mask=mask)
    # print(res)
    # cv2.imwrite(pic[:-4]+'_red.jpg',res)
    return res


def batch_process_on_hsv():
    img_dir = './crop_data/images'
    output_dir = './crop_data/images_hsv'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    img_fn_list = os.listdir(img_dir)
    for img_fn in img_fn_list:
        if not (img_fn.endswith('.jpg') or img_fn.endswith('.png')):
            continue
        # print("---------------------------------------------------------------------")
        # since = time.time()
        img_path = os.path.join(img_dir, img_fn)
        # print(img_path)
        img_ori = cv2.imread(img_path)
        hsv_img = res_area(img_ori)
        cv2.imwrite(os.path.join(output_dir, img_fn.replace('.jpg', '_res.jpg')), hsv_img)


def image_preprocess_by_normality(img_cv2, seq_type):
    # scene BGR: [32.26, 30.01, 30.89], std: 37.76
    # hsv BGR: [2.428, 2.572, 4.687], std: 16.12
    '''  # HSV
    if seq_type == 'RGB':
        mean = (4.687, 2.572, 2.428)

    else:  # BGR
        mean = (2.428, 2.572, 4.687)
    std = 16.12
    '''
    # scene
    if seq_type == 'RGB':
        mean = (30.89, 30.01, 32.26)
    
    else:  # BGR
        mean = (32.26, 30.01, 30.89)
    std = 37.76

    img_data = np.asarray(img_cv2, dtype=np.float32)
    img_data = img_data - mean
    img_data = img_data / std
    img_data = img_data.astype(np.float32)
    return img_data


def calc_image_mean_and_std():
    img_dir = '/data/data/weche_train_data/images'
    #img_dir = './source_data/images_hsv'

    mean_B_list = []
    mean_G_list = []
    mean_R_list = []
    std_list = []

    img_fn_list = os.listdir(img_dir)[0:2000]
    for img_fn in img_fn_list:
        if not (img_fn.endswith('.jpg') or img_fn.endswith('.png')):
            continue
        print("---------------------------------------------------------------------")
        print('img_fn: ', img_fn)
        img_path = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_path)
        img_data = np.asarray(img_cv2, dtype=np.float)

        b_layer = img_data[:, :, 0]
        g_layer = img_data[:, :, 1]
        r_layer = img_data[:, :, 2]

        mean_b = np.mean(b_layer)
        mean_g = np.mean(g_layer)
        mean_r = np.mean(r_layer)
        std = np.std(img_data)

        print('B/G/R/std: ', mean_b, mean_g, mean_r, std)
        mean_B_list.append(mean_b)
        mean_G_list.append(mean_g)
        mean_R_list.append(mean_r)
        std_list.append(std)

    print('---------------- calc mean ----------------')
    mean_B = np.mean(mean_B_list)
    mean_G = np.mean(mean_G_list)
    mean_R = np.mean(mean_R_list)
    g_std = np.mean(std_list)
    print('mean_BGR: ', mean_B, mean_G, mean_R)
    print('std: ', g_std)


if __name__ == '__main__':
    
    calc_image_mean_and_std()   
    '''
    a = [[1, 2, 3], [4, 5, 6]]
    res = image_preprocess_by_normality(a, seq_type='RGB')
    print(res)
    res = res.astype(np.float32)
    print(res)
    '''

