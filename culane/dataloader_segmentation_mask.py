import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


"""
Based of https://github.com/cardwing/Codes-for-Lane-Detection/blob/master/ERFNet-CULane-PyTorch/dataset/voc_aug.py 
"""

# TODO: not yet used
LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}

# TODO: not yet used
CATEGORIES = {
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
}


class CULaneDataset(Dataset):
    def __init__(self, dataset_path='/Users/charlesdowns/Documents/GitHub/IAAIP2023/culane/data/list', data_list='train_gt', transform=None):
        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            self.spline_coefs = []
            # read all lines in file and fill in the lists, nb: not all are currently used
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform
        self.is_testing = data_list == 'test_img' # 'val' # update to be one of the keys in LIST_FILE

        

    def __len__(self):
        return len(self.img_list)
    
    def read_annotation(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        coords = []
        for line in lines:
            coords.append([int(line.strip().split(' ')[0]), int(line.strip().split(' ')[1])])
        return coords


    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.uint8)  # Use uint8 data type
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB order
        label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
        exist = self.exist_list[idx]
        label = label.squeeze()
        if self.transform: # need to define own Transform classes here if we want to do any preprocessing
            image, label = self.transform((image, label))
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
            label = torch.from_numpy(label).contiguous().long()
        if self.is_testing:
            return image, label, self.img[idx]
        else:
            return image, label, exist