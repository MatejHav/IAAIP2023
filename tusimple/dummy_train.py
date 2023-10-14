import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import numpy as np
import dataloader as ds
import torchvision 
import torchvision.transforms
import dataloader as ds

SPLIT_FILES = {
    'trainval':
    ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

if __name__ == '__main__':
    batch_size = 8    
    anno_files = SPLIT_FILES['train']
    # dataset_path = 'C:/Users/Charles/Documents/GitHub/IAAIP-Transformer/tusimple/data/TUSimple/train_set'
    dataset_path = '/home/charles/Desktop/IAAIP-Transformer/tusimple/data/TUSimple/train_set'

    dataset = ds.TUSimpleDataset(dataset_path, anno_files)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    
    print(f"Length of dataset: {len(dataset)}")

    for i, (image, label, lanes, img_path) in enumerate(train_loader):
        image, label, lanes, img_path = image[0], label[0], lanes[0], img_path[0]
        # broken 