### crop raw data into patches
import os
import cv2
import natsort
from skimage import io


if __name__ == "__main__":
    if __name__ == "__main__":
        # crop images
        image_path = '/data/zhenghui.zhao/Dataset/Change Detection/AICD/Images_NoShadow/A'
        save_image_path = '/data/zhenghui.zhao/Dataset/Change Detection/AICD_noshadow_256/A'
        # crop labels
        label_path = '/data/zhenghui.zhao/Dataset/Change Detection/AICD/GroundTruth'
        save_label_path = '/data/zhenghui.zhao/Dataset/Change Detection/AICD_noshadow_256/label'

        file_list = os.listdir(image_path)
        inList = natsort.natsorted(file_list)
        file_list2 = os.listdir(label_path)
        inList2 = natsort.natsorted(file_list2)

        num = 1
        imgsize = 256

        for name in inList:

            image = io.imread(image_path + '/' + name)
            img = cv2.resize(image, (768,512))
            try:
                img=img[:, :, :4]
                img=img[:,:,:3]
                print(name)
            except:
                img = img
            crop1 = img[:imgsize, :imgsize, :]
            crop2 = img[imgsize:2 * imgsize, :imgsize, :]
            crop3 = img[:imgsize, imgsize:2 * imgsize, :]
            crop4 = img[imgsize:2 * imgsize, imgsize:2 * imgsize, :]
            crop5 = img[:imgsize, 2*imgsize: 3*imgsize, :]
            crop6 = img[imgsize:2 * imgsize, 2*imgsize:3*imgsize, :]

            print('crop1:', crop1.shape)
            print('crop2:', crop2.shape)
            print('crop3:', crop3.shape)
            print('crop4:', crop4.shape)
            print('crop5:', crop5.shape)
            print('crop6:', crop6.shape)

            #print(save_image_path + '/' + name[:17] + 'crop1' + name[-4:])
            #print(save_label_path + '/' + name[:17] + 'crop1_' + name[-4:])
            io.imsave(save_image_path + '/' + name[:16] + '_crop1' + name[-4:], crop1)
            io.imsave(save_image_path + '/' + name[:16] + '_crop2' + name[-4:], crop2)
            io.imsave(save_image_path + '/' + name[:16] + '_crop3' + name[-4:], crop3)
            io.imsave(save_image_path + '/' + name[:16] + '_crop4' + name[-4:], crop4)
            io.imsave(save_image_path + '/' + name[:16] + '_crop5' + name[-4:], crop5)
            io.imsave(save_image_path + '/' + name[:16] + '_crop6' + name[-4:], crop6)

        for name in inList2:

            print(name[:17] + 'crop1'+name[-4:])

            img = io.imread(label_path + '/' + name)
            img = cv2.resize(img, (768,512))
            try:
                img = img[:,:,0]
                print(name)
            except:
                img = img
            crop1 = img[:imgsize, :imgsize]
            crop2 = img[imgsize:2 * imgsize, :imgsize]
            crop3 = img[:imgsize, imgsize:2 * imgsize]
            crop4 = img[imgsize:2 * imgsize, imgsize:2 * imgsize]
            crop5 = img[:imgsize, 2 * imgsize:3 * imgsize]
            crop6 = img[imgsize:2 * imgsize, 2 * imgsize:3 * imgsize]
            
            print('crop1:', crop1.shape)
            print('crop2:', crop2.shape)
            print('crop3:', crop3.shape)
            print('crop4:', crop4.shape)
            print('crop5:', crop5.shape)
            print('crop6:', crop6.shape)
            
            io.imsave(save_label_path + '/' + name[:17] + 'crop1'+ name[-4:], crop1)
            io.imsave(save_label_path + '/' + name[:17] + 'crop2'+name[-4:], crop2)
            io.imsave(save_label_path + '/' + name[:17] + 'crop3'+name[-4:], crop3)
            io.imsave(save_label_path + '/' + name[:17] + 'crop4'+name[-4:], crop4)
            io.imsave(save_label_path + '/' + name[:17] + 'crop5'+name[-4:], crop5)
            io.imsave(save_label_path + '/' + name[:17] + 'crop6'+name[-4:], crop6)

        '''for name in inList:
            #print(name)  # T1, T2
            #print(name[:17] + 'crop1')

            img = io.imread(image_path+'/'+name)
            #print(img.shape)
            try:
                img = img[:,:,0]
                print(name)
            except:
                img = img
            print(img.shape)
            io.imsave(save_image_path + '/' + name, img)'''



