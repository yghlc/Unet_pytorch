import torch.utils.data as data
import torch

from scipy.ndimage import imread
import os
import os.path
import glob

import numpy as np

HOME = os.path.expanduser('~')
import sys

basicCodes_path = HOME + '/codes/PycharmProjects/DeeplabforRS'
sys.path.insert(0, basicCodes_path)
import split_image


class patch(object):
    """
    store the information of each patch (a small subset of the remote sensing images)
    """
    def __init__(self,org_img):
        self.org_img = org_img  # the original remote sensing images of this patch (the name of image without absolute path)
        self.boundary=None      # the boundary of patch (xoff,yoff ,xsize, ysize) in pixel coordinate


def check_input_image_and_label():
    """
    check the input image and label, they should have same width, height, and projection 
    :return: (width, height) of image if successful, Otherwise None.
    """

    #TODO: add codes to check the image and lable (21 Nov, 2017, hlc)
    pass


def make_dataset(root,list_txt, train=True):
    """
    get the patches information of the remote sensing images. 
    :param root: data root
    :param list_txt: a list file contain images (one row contain one train image and one label 
    image with space in center if input for training; one row contain one image if it is for inference)
    :param train:  indicate training or inference
    :return: dataset (list)
    """
    dataset = []

    if os.path.isfile(list_txt) is False:
        print("error, file %s not exist"%list_txt)
        assert False

    with open(list_txt) as file_obj:
        files_list = file_obj.readlines()
    if len(files_list) < 1:
        print("error, no file name in the %s" % list_txt)
        assert False

    if train:
        for line in files_list:
            names_list = line.split()
            image_name = names_list[0]
            label_name = names_list[1].strip()

            #
            (width,height) = check_input_image_and_label()

            # split the image and label
            split_image.sliding_window()




        dataset.append([fImg, fGT])
    else:
        image_dir = os.path.join(root, 'test_img')
        dataset = glob.glob(os.path.join(image_dir, '*.tif'))
    #    for img in glob.glob(os.path.join(image_dir, '*.tif')):
    #      dataset.append([img])

    return dataset


def getImg_count(dir):
    files = glob.glob(os.path.join(dir, '*.tif'))
    return len(files)


class RemoteSensingImg(data.Dataset):
    """
    Read dataset of kaggle ultrasound nerve segmentation dataset
    https://www.kaggle.com/c/ultrasound-nerve-segmentation
    """

    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.root = root

        # we cropped the image(the size of each patch, can be divided by 16 )
        self.nRow = 480
        self.nCol = 480

        self.train_set_path = make_dataset(root, train)

    def __getitem__(self, idx):
        if self.train:
            img_path, gt_path = self.train_set_path[idx]

            img = imread(img_path)
            # img.resize(self.nRow,self.nCol)
            img = img[0:self.nRow, 0:self.nCol]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            if (img.max() - img.min()) < 0.01:
                pass
            else:
                img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()

            gt = imread(gt_path)[0:self.nRow, 0:self.nCol]
            gt = np.atleast_3d(gt).transpose(2, 0, 1)
            # gt = gt / 255.0   # we don't need to scale
            gt = torch.from_numpy(gt).float()

            return img, gt
        else:
            img_path = self.train_set_path[idx]
            img_name_noext = os.path.splitext(os.path.basename(img_path))[0]
            img = imread(img_path)
            # img.resize(self.nRow,self.nCol)
            img = img[0:self.nRow, 0:self.nCol]
            img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            if (img.max() - img.min()) < 0.01:
                pass
            else:
                img = (img - img.min()) / (img.max() - img.min())
            img = torch.from_numpy(img).float()
            return img, img_name_noext

    def __len__(self):
        if self.train:
            # train image count
            label_dir = os.path.join(self.root, 'split_labels')
            count = getImg_count(label_dir)
            print("Image count for training is %d" % count)
            return count
            # return 5635
            # test image count
        else:
            label_dir = os.path.join(self.root, 'inf_split_images')
            count = getImg_count(label_dir)
            print("Image count for inference is %d" % count)
            return count
            # return 5508   # test image count
