import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CULaneDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_paths, self.anno_paths = self._load_data_paths()
        self.normalize = True  # This can be made an argument or flag
        self.img_w, self.img_h = 1640, 590  # Default image dimensions for CULane
        
    def _load_data_paths(self):
        # Assumes the structure is:
        # - root
        #   - driver_23_30frame
        #     - xxxxxxxx.jpg
        #     - xxxxxxxx.lines.txt
        image_paths = []
        anno_paths = []
        
        for subdir, _, files in os.walk(self.root):
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(subdir, file))
                    anno_path = os.path.join(subdir, file.replace('.jpg', '.lines.txt'))
                    anno_paths.append(anno_path)
                    
        return image_paths, anno_paths

    def _load_annotation(self, anno_path):
        with open(anno_path, 'r') as lanes_obj:
            strlanes = lanes_obj.readlines()
            
        lanes = []
        for strlane in strlanes:
            strpts = strlane.split(' ')[:-1]
            x_gts = [float(x_) for x_ in strpts[::2]]
            y_gts = [float(y_) for y_ in strpts[1::2]]
            lanes.append([(x, y) for (x, y) in zip(x_gts, y_gts) if x >= 0])
        
        return lanes

    def __len__(self):
        return len(self.img_paths)
    
    def lane_to_img(self, lanes):
        # Convert lanes to image
        img = np.zeros((self.img_h, self.img_w))
        # use polylines
        for lane in lanes:
            pts = np.array(lane, np.int32)
            pts = pts.reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts], False, 255)

        return img

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        anno_path = self.anno_paths[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        lanes = self._load_annotation(anno_path)
        
        if self.transform:
            # This part assumes you have some transformation function
            img, lanes = self.transform(img, lanes)
            
        if self.normalize:
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # ImageNet mean and std
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        
        return img_tensor, lanes

# Usage
# dataset = CULaneDataset("/Users/charlesdowns/Documents/GitHub/IAAIP2023/culane/data/driver_23_30frame") # NOTE: = subset of image dataset!
# img_tensor, lanes = dataset[0]
# print(img_tensor.shape, lanes)

# # display image and lanes
# import matplotlib.pyplot as plt
# # convert the img_tensor to proper RGB image
# img_tensor = img_tensor.numpy().transpose(1, 2, 0)
# img_tensor = img_tensor * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
# img_tensor = img_tensor * 255
# img_tensor = img_tensor.astype(np.uint8)
# plt.imshow(img_tensor)
# for lane in lanes:
#     plt.plot([x for (x, y) in lane], [y for (x, y) in lane])
# plt.show()

# lanes_image = dataset.lane_to_img(lanes)
# plt.imshow(lanes_image)
# plt.show()