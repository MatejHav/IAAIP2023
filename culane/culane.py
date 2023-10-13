import os
import pickle
import logging

import cv2
import numpy as np
from tqdm import tqdm
import torch

import culane.utils.culane_metric as culane_metric

from culane.lane_dataset_loader import LaneDatasetLoader

SPLIT_FILES = {
    'train': "list/train.txt",
    'val': 'list/val.txt',
    'test': "list/test.txt",
    'normal': 'list/test_split/test0_normal.txt',
    'crowd': 'list/test_split/test1_crowd.txt',
    'hlight': 'list/test_split/test2_hlight.txt',
    'shadow': 'list/test_split/test3_shadow.txt',
    'noline': 'list/test_split/test4_noline.txt',
    'arrow': 'list/test_split/test5_arrow.txt',
    'curve': 'list/test_split/test6_curve.txt',
    'cross': 'list/test_split/test7_cross.txt',
    'night': 'list/test_split/test8_night.txt',
    'debug': 'list/debug.txt'
}


class CULane(LaneDatasetLoader):
    def __init__(self, max_lanes=None, split='train', root=None, official_metric=True, save_formatted=False, save_fit=False, load_formatted=True, load_fit=False, subset=100):
        self.split = split
        self.root = root
        self.official_metric = official_metric
        self.load_formatted = load_formatted
        self.load_fit = load_fit
        self.save_formatted = save_formatted
        self.save_fit = save_fit
        self.logger = logging.getLogger(__name__)
        SPLIT_FILES['train'] = SPLIT_FILES['train'][:-4] + ('_' + str(subset) if subset < 100 else '') + '.txt'

        if root is None:
            raise Exception('Please specify the root directory')
        if split not in SPLIT_FILES:
            raise Exception('Split `{}` does not exist.'.format(split))

        self.list = os.path.join(root, SPLIT_FILES[split])

        self.img_w, self.img_h = 1640, 590
        self.subset = subset
        self.annotations = []
        self.load_annotations()
        self.max_lanes = 4 if max_lanes is None else max_lanes

    def get_img_heigth(self, _):
        return self.img_h

    def get_img_width(self, _):
        return self.img_w

    def load_annotation(self, img_path):
        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt

        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]

        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) >= 2]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y

        if self.load_formatted and not self.save_formatted:
            return {'path': img_path, 'lanes': torch.load(img_path[:-4] + '_lines_formatted.txt'), 'old_lanes': lanes}
        if self.load_fit and not self.save_fit:
            return {'path': img_path, 'lanes': torch.load(img_path[:-4] + '_lines_fit.txt'), 'old_lanes': lanes}

        return {'path': img_path, 'lanes': lanes, 'old_lanes': lanes}

    def create_mask(self, lanes):
        mask = np.zeros((self.img_h, self.img_w, 3))

        for lane in lanes:
            for index in range(1, len(lane)):
                cv2.line(mask, list(map(int, lane[index - 1])), list(map(int, lane[index])), color=(1, 1, 1), thickness=3)
        return mask[:, :, 0].astype(np.float32)

    def load_annotations(self):
        self.annotations = []
        self.max_lanes = 0
        os.makedirs('./culane/cache', exist_ok=True)
        cache_path = f'./culane/cache/culane_{self.split}_{"formatted" if self.load_formatted else "raw"}' \
                     + ('_' + str(self.subset) if self.subset < 100 else '')

        if os.path.exists(cache_path):
            # print(f'LOADING {self.split.upper()} CACHED DATA')
            with open(cache_path, 'rb') as cache_file:
                data = pickle.load(cache_file)
                self.annotations = data['annotations']
        else:
            # print(f'LOADING {self.split.upper()} VIDEOS INTO CACHE')
            with open(self.list, 'r') as list_file:
                files = [line.rstrip()[1 if line[0] == '/' else 0::]
                         for line in list_file]  # remove `/` from beginning if needed

            bar = tqdm(files)
            for file in bar:
                bar.set_description(f'Loading video {file.split("/")[-3]}/{file.split("/")[-2]}')
                img_path = os.path.join(self.root, file)
                anno = self.load_annotation(img_path)
                anno['org_path'] = file
                self.annotations.append(anno)
            with open(cache_path, 'wb') as cache_file:
                pickle.dump({'annotations': self.annotations}, cache_file)

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
