import numpy as np
from numpy.lib.utils import deprecate
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import imageio
from . import transforms_CD as transforms

"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_cls_label_list(name_list_dir):
    return np.load(os.path.join(name_list_dir, 'imagelevel_labels.npy'), allow_pickle=True).item()


class weaklyCDDataset(Dataset):
    def __init__(
            self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
            img_size=256,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir_A = os.path.join(root_dir, IMG_FOLDER_NAME)
        self.img_dir_B = os.path.join(root_dir, IMG_POST_FOLDER_NAME)
        self.label_dir = os.path.join(root_dir, ANNOT_FOLDER_NAME)
        self.name_list_dir = os.path.join(name_list_dir,split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)
        self.A_size = len(self.name_list)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name_A = os.path.join(self.img_dir_A, self.name_list[idx % self.A_size])
        img_name_B = os.path.join(self.img_dir_B, self.name_list[idx % self.A_size])
        image_A = np.asarray(imageio.imread(img_name_A))
        image_B = np.asarray(imageio.imread(img_name_B))


        if self.stage == "train":

            label_dir = os.path.join(self.label_dir, _img_name)
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "val":

            label_dir = os.path.join(self.label_dir, _img_name)
            label = np.asarray(imageio.imread(label_dir))

        elif self.stage == "test":   #
            label_dir = os.path.join(self.label_dir, _img_name)
            label = np.asarray(imageio.imread(label_dir))

        return _img_name, image_A, image_B, label


class ClsDataset(weaklyCDDataset):         # train_dataset
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 rescale_range=[0.5, 2.0],
                 crop_size=None,
                 img_fliplr=True,
                 random_color_tf = False,
                 ignore_index=255,
                 num_classes=2,
                 aug=False,
                 **kwargs):
###
        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.random_color_tf = random_color_tf
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image_A, image_B):
        img_box = None

        if self.aug:
            if self.rescale_range:
                image_A, image_B = transforms.random_scaling(
                    image_A, image_B,
                    scale_range=self.rescale_range)

            if self.img_fliplr:
                image_A, image_B = transforms.random_fliplr(image_A, image_B)

            # image_A = self.color_jittor(image_A)
            # image_B = self.color_jittor(image_B)

            ### 弱监督变化检测，不能用随机裁剪 ###
            if self.crop_size:
                image_A, image_B, img_box = transforms.random_crop(
                    image_A, image_B,
                    crop_size=self.crop_size,
                    mean_rgb=[0, 0, 0],  # [123.675, 116.28, 103.53],
                    )
            ##################################

        image_A = transforms.normalize_img(image_A)
        image_A = np.transpose(image_A, (2, 0, 1))
        image_B = transforms.normalize_img(image_B)
        image_B = np.transpose(image_B, (2, 0, 1))

        return image_A, image_B, img_box


    def __getitem__(self, idx):   # img_box for transformational samples

        img_name, image_A, image_B, _ = super().__getitem__(idx)

        image_A, image_B, img_box = self.__transforms(image_A=image_A, image_B=image_B)


        cls_label = self.label_list[img_name]

        if self.aug:
            return img_name, image_A, image_B, cls_label, img_box
        else:
            return img_name, image_A, image_B, cls_label


class CDDataset(weaklyCDDataset):      # val_dataset
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image_A, image_B, label):
        if self.aug:
            if self.img_fliplr:
                image_A, image_B, label = transforms.random_fliplr(image_A, image_B, label)
            if self.crop_size:
                image_A, image_B, label = transforms.random_crop(
                    image_A, image_B,
                    label,
                    crop_size=self.crop_size,
                    mean_rgb=[123.675, 116.28, 103.53],
                    )

        image_A = transforms.normalize_img(image_A)
        image_B = transforms.normalize_img(image_B)
        image_A = np.transpose(image_A, (2, 0, 1))
        image_B = np.transpose(image_B, (2, 0, 1))


        return image_A, image_B, label

    def __getitem__(self, idx):
        img_name, image_A, image_B, label = super().__getitem__(idx)

        image_A, image_B, label = self.__transforms(image_A=image_A, image_B=image_B, label=label)

        cls_label = self.label_list[img_name]

        label = label // 255

        return img_name, image_A, image_B, label, cls_label
