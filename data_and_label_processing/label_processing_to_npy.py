# -*- coding: utf-8 -*#
# pixel-level  ——> image-level labels

import os
import numpy as np
import cv2
import natsort



if __name__ == "__main__":
    file_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256/label'
    save_path = '/data/zhenghui.zhao/Dataset/Change Detection/WHU-CD-256'
    file_list = os.listdir(file_path)
    inList = natsort.natsorted(file_list)


    num = 0
    n = 0
    m = 0
    image_dict = {}
    #change = np.array([0., 0.], dtype=np.uint8)
    for name in inList:
        img = cv2.imread(file_path+'/'+ name)
        if img.any():
            change = np.array([1], dtype=np.float32)
            m = m+1
        else:
            change =  np.array([0], dtype=np.float32)
            n = n+1
        image_dict[name] = change

        num = num+1
    print('label_num:', num)
    print('nonchange_num:', n)
    print('change_num:', m)
    #print(image_dict)

    np.save(save_path+'/imagelevel_labels.npy', image_dict)
    #np.save('/data/zhenghui.zhao/Code/Affinity-from-attention/Affinity-from-attention-transformer/dual_stream/datasets/AICD_128/imagelevel_labels.npy', image_dict)
    #np.save('/data/zhenghui.zhao/Code/Affinity-from-attention/Affinity-from-attention-transformer_Single/dual_stream/datasets/AICD_128/imagelevel_labels.npy',image_dict)

    label = np.load(save_path+'/imagelevel_labels.npy',allow_pickle=True)
    #test = np.load('./dual_stream/datasets/voc/cls_labels_onehot.npy', allow_pickle=True)
    print(label)
    print('save_image_path:', save_path+'/imagelevel_labels.npy')


