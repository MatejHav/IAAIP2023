import os
import numpy as np
import json
import cv2
import torch
import os.path as osp
import random
from torch.utils.data import Dataset

SPLIT_FILES = {
    'trainval':
    ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

class TUSimpleDataset(Dataset):
    def __init__(self, dataset_path, anno_files):
        self.data_infos = []
        self.anno_files = anno_files
        self.dataset_path = dataset_path
        self.h_samples = list(range(160, 720, 10)) # not sure what this does
        max_lanes = 0
        for anno_file in self.anno_files:
            # keep it here for now, since we want to open each file and then load data - no need to re-open the file each time we read a line :) 
            anno_file = osp.join(dataset_path, anno_file)
            with open(osp.join(dataset_path, anno_file)) as f:
                for line in f:
                    data = json.loads(line)
                    y_samples = data['h_samples']
                    gt_lanes = data['lanes']
                    mask_path = data['raw_file'].replace('clips',
                                                     'seg_label')[:-3] + 'png'
                    
                    # redundant, not used, but could be used for polylines later. 
                    lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                            for lane in gt_lanes]
                    lanes = [lane for lane in lanes if len(lane) > 0]
                    max_lanes = max(max_lanes, len(lanes))

                    # append all info to the data_infos dict. 
                    self.data_infos.append({
                        'img_path':
                        osp.join(self.dataset_path, data['raw_file']),
                        'img_name':
                        data['raw_file'],
                        'mask_path':
                        osp.join(self.dataset_path, mask_path),
                        'lanes':
                        lanes,
                    })

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.dataset_path, self.data_infos[idx]['img_path'])).astype(np.uint8)  # Use uint8 data type
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(self.dataset_path, self.data_infos[idx]['mask_path']), cv2.IMREAD_UNCHANGED)
        lanes = self.data_infos[idx]['lanes']
        clip_path = self.data_infos[idx]['img_path']

        # TODO: could use the lanes (list of coords) to create  polylines via cv2. 
        # TODO: apply transforms if self.transform == True

        return image, label, lanes, clip_path
    
    def getname(self, idx):
        return self.data_infos[idx]['img_name']
    
    def get_clip_path(self, idx):
        return os.path.dirname(self.data_infos[idx]['img_path'])
    
    def get_clips_Stack(self, idx):
        clips_stack = []
        clip_dir = self.get_clip_path(idx)
        
        # Check if the directory exists
        if os.path.exists(clip_dir):
            for i in range(1, 21):  # Assuming files are named from 1.jpg to 20.jpg
                file_name = f"{i}.jpg"
                file_path = os.path.join(clip_dir, file_name)
                if os.path.isfile(file_path):
                    image = cv2.imread(file_path)  # Read the image using OpenCV
                    if image is not None:
                        clips_stack.append(image)
        else:
            print(f"Directory not found: {clip_dir}")

        return clips_stack


# main
if __name__ == '__main__':
    anno_files = SPLIT_FILES['train']
    # dataset_path = 'C:/Users/Charles/Documents/GitHub/IAAIP-Transformer/tusimple/data/TUSimple/train_set'
    dataset_path = '/home/charles/Desktop/IAAIP-Transformer/tusimple/data/TUSimple/train_set'
    dataset = TUSimpleDataset(dataset_path, anno_files)

    for i in range (0, len(dataset)):
        image, label, lanes, img_path = dataset[i]
        img_name = dataset.getname(i)
        clip_stack = dataset.get_clip_path(i)
        print('clip path = ', clip_stack)

        # convert to binary mask (should already be, but just in case)
        binary_mask = (label > 0).astype(np.uint8) 
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2RGB)
        # streth values to [0, 255] range 
        binary_mask = binary_mask * 255

        # display the raw mask
        cv2.imshow('label', binary_mask)
        cv2.waitKey(0)

        # create polylines as an empty cv2 image
        rows, cols, channels = image.shape
        polylines = np.zeros((rows, cols, channels), dtype=np.uint8)
        for lane in lanes:
            for point in lane:
                # cv2.circle(binary_mask, point, 2, (255, 0, 0), -1)
                cv2.polylines(polylines, np.int32([lane]), isClosed=False, color=(255, 0, 0), thickness=2)

        # since there's no easy way of getting each 20 image from a clip, we need to make a manual fuction to do this....
        clip_stack = dataset.get_clips_Stack(i)
        for clip in clip_stack:
            # overlay the label image on top of the image
             overlayed_image = cv2.addWeighted(polylines, 0.5, clip, 1, 0)
             cv2.imshow('clip frame', overlayed_image)
             cv2.waitKey(0)
