import os
import pickle
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from imgaug.augmenters import Resize
from imgaug.augmentables.lines import LineString, LineStringsOnImage

class CULaneDataset(Dataset):

    def __init__(self, root_dir, split):
        # Directory information
        self.root_dir = root_dir
        self.split = split
        self.root = root_dir
        self.anno_files = [os.path.join(self.root, 'list', file_name) for file_name in split]

        # Image dimensions
        self.img_w, self.img_h = 1640, 590
        
        # Load data and preprocess
        self._annotations = {}
        self._image_ids = []
        self.max_points = 0
        self._load_data()

        # Transformations
        # self.transform = Resize({'height': self.img_h, 'width': self.img_w})
        self.transform = None
        self.to_tensor = ToTensor()

    def _pad_to_length(self, lane, max_length, pad_value=(-1e5, -1e5)):
        return lane + [pad_value] * (max_length - len(lane))

    def _load_data(self):
        image_id = 0
        for anno_file in self.anno_files:
            with open(anno_file, 'r') as anno_obj:
                annolines = anno_obj.readlines()
            for annoline in annolines:
                rel_path = annoline.strip()
                img_path = os.path.join(self.root_dir, rel_path)
                anno_path = os.path.join(self.root_dir, rel_path[:-4] + '.lines.txt')
                # print('checking for file: {}'.format(anno_path))
                with open(self.root_dir + '' + anno_path, 'r') as lanes_obj:
                    strlanes = lanes_obj.readlines()
                lanes = []
                for strlane in strlanes:
                    strpts = strlane.split(' ')[:-1]
                    y_gts = [float(y_) for y_ in strpts[1::2]]
                    x_gts = [float(x_) for x_ in strpts[::2]]
                    lane = [(x, y) for (x, y) in zip(x_gts, y_gts) if x >= 0]
                    lanes.append(lane)

                if not lanes:
                    continue

            
                lanes = [lane for lane in lanes if len(lane) > 0]
                self.max_points = max(self.max_points, max([len(l) for l in lanes]))

                # Pad lanes to a consistent length
                lanes = [self._pad_to_length(lane, self.max_points) for lane in lanes]

                self._image_ids.append(image_id)
                self._annotations[image_id] = {
                    'path': self.root_dir + '' + img_path,
                    'lanes': lanes
                }
                image_id += 1

                if lanes:
                    self.max_points = max(self.max_points, max([len(l) for l in lanes]))

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, idx):
        item = self._annotations[idx]
        img = cv2.imread(item['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        lanes = item['lanes']

        if self.transform:
            img = self.transform(image=img)['image']
        img = self.to_tensor(img)

        return img, lanes