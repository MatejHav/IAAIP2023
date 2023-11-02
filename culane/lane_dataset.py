import math

import cv2
import numpy as np
import imgaug.augmenters as iaa
import torch
from PIL import Image
from imgaug.augmenters import Resize
from timm.models.vision_transformer import PatchEmbed
from torch import nn
from torchvision import transforms
import torchvision.transforms as T
from scipy import ndimage
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset

from culane.culane import CULane

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
GROUND_TRUTH_GRID = (64, 160)

# Based on: https://github.com/lucastabelini/LaneATT/tree/2f8583ba14eccba05e6779668bc3a38bc751984a
#

def adjust_ground_truth(img, item, augmentation, angle=None):
    lanes = item['lanes']
    for lane in lanes:
        for i in range(len(lane)):
            match augmentation:
                case 'flip':
                    lane[i] = (img.shape[1] - lane[i][0], lane[i][1])
                case _:
                    raise Exception


class LaneDataset(Dataset):

    def __init__(self,
                 dataset='culane',
                 normalize=False,
                 img_size=(550, 550),
                 resize_size=(224, 224),
                 **kwargs):
        super(LaneDataset, self).__init__()
        if dataset == 'culane':
            self.dataset = CULane(**kwargs)
        else:
            raise NotImplementedError()
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.resized_h, self.resized_w = resize_size
        self.resize = T.Resize(resize_size)
        self.to_tensor = ToTensor()
        self.augmentations = {}

    @property
    def annotations(self):
        return self.dataset.annotations

    # Method2 - Translating the point to a new coordinate system with the point of rotation being the origin

    def transform_annotations(self):
        """
        Transform dataset annotations to the model's target format.
        """
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))

    def filter_lane(self, lane):
        """
        Filters lane points, removing those with duplicate y-values.
        
        Args:
            lane (list): List of [x, y] coordinates representing the lane.
            
        Returns:
            list: Filtered lane with unique y-values.
        """
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def create_mask(self, lanes):
        mask = np.zeros((self.img_h, self.img_w, 3))

        y, x = np.where(lanes[:, :, 2] >= 0.5)

        length = lanes[y, x, 0]
        angle = lanes[y, x, 1]

        x1 = (x / lanes.shape[1] * mask.shape[1]).astype(int)
        y1 = (y / lanes.shape[0] * mask.shape[0]).astype(int)

        x2 = ((x / lanes.shape[1] + np.cos(angle) * length) * mask.shape[1]).astype(int)
        y2 = ((y / lanes.shape[0] + np.sin(angle) * length) * mask.shape[0]).astype(int)

        temp = np.column_stack((x1, y1, x2, y2))

        for x1, y1, x2, y2 in temp:
            cv2.line(mask, (x1, y1), (x2, y2), color=(1, 1, 1), thickness=3)

        return mask[:, :, 0].astype(np.float32)

    def apply_augmentations(self, video_path, img, mask):
        y, x = self.augmentations[video_path][0]
        rotation = self.augmentations[video_path][1]
        flip = self.augmentations[video_path][2]
        img = rotation(flip(img))
        mask = rotation(flip(mask.unsqueeze(0)))
        img = img[:, y:y + self.img_h, x:x + self.img_w]
        mask = mask[:, y:y + self.img_h, x:x + self.img_w]
        img = self.resize(img)
        mask = self.resize(mask).squeeze(0)
        return img, mask

    def __getitem__(self, idx):
        item = self.dataset[idx]

        img = cv2.imread(item['path'])
        mask = self.dataset.create_mask(item['lanes'])

        # Standardize image
        img = img / 255
        img = np.array(img)
        # Normalize image
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # Convert to tensor
        img = self.to_tensor(img.astype(np.float32))
        mask = torch.Tensor(mask)

        # Resize image
        video_path = item['path'].split('/')[-2]
        if video_path not in self.augmentations:
            y = np.random.randint(0, img.shape[1] - self.img_h)
            x = np.random.randint(0, img.shape[2] - self.img_w)
            degree = np.random.randint(-15, 15)
            p = 0 if np.random.random() < 0.5 else 1
            self.augmentations[video_path] = [
                (y, x),
                T.RandomRotation(degrees=(degree-2, degree+2)),
                T.RandomHorizontalFlip(p=p)
            ]
        img, mask = self.apply_augmentations(video_path, img, mask)

        return img, mask

    def __len__(self):
        return len(self.dataset)
