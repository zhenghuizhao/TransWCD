# rename dir for T1, T2 & pixel-level label
import os

import natsort
from shutil import copy

if __name__ == "__main__":
    file_path = '/data/zhenghui.zhao/Dataset/Change Detection/AICD/Images_NoShadow'
    file_path2 = '/data/zhenghui.zhao/Dataset/Change Detection/AICD_noshadow_256/A'
    file_list_1 = os.listdir(file_path)
    file_list_2 = os.listdir(file_path2)
    inList_1 = natsort.natsorted(file_list_1)
    inList_2 = natsort.natsorted(file_list_2)


    for item in inList_1:
        if '_target' in item:
                    print('Source：',file_path+'/'+item)
                    item_new = item.replace('_target', '')
                    print('Target：', file_path + '/A/' + item_new)
                    copy(file_path+'/'+item, file_path+'/A/'+item_new)


        '''if 'mov' in item:
                    print('Source：',image_path+'/'+item)
                    item_new = item.replace('_moving', '')
                    print('Target：', image_path + '/B/' + item_new)
                    #os.rename(os.path.join(image_path, item), os.path.join(image_path, item_new))
                    copy(image_path+'/'+item, image_path+'/B/'+item_new)'''



    '''for item in inList_1:
        #item_new = item.replace('_gtmask', '')
        #item_new = item.replace('_target', '')
        print('Source：',image_path+'/'+item)
        copy(image_path+'/'+item, image_path+'/A/'+item)
        print('Target：', image_path+'/A/'+item)'''

    '''for item in inList_2:
        #item_new = item.replace('_gtmask', '')
        #item_new = item.replace('_target', '')
        item_new = item.replace('_moving', '')
        print('Source：',item)
        os.rename(os.path.join(label_path, item), os.path.join(label_path,item_new))  
        print(item, "has been renamed successfully! New name is: ", item_new)  '''
