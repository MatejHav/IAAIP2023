import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
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

class LaneDataset(Dataset):
    """
    A PyTorch dataset class to handle lane detection data. 
    It abstracts the data loading, preprocessing, augmentation, and conversion to model-friendly formats.

    Args:
            S (int): Number of strips.
            dataset (str): Dataset type ('culane' supported).
            augmentations (list): List of data augmentation configurations.
            normalize (bool): Whether to normalize image data.
            img_size (tuple): Tuple representing the image size (height, width).
            aug_chance (float): Probability of applying augmentations.
            **kwargs: Additional keyword arguments.

    ttributes:
            annotations: List of dataset annotations.
            n_strips (int): Number of strips.
            n_offsets (int): Number of offsets.
            strip_size (float): Size of each strip.
            img_h (int): Image height.
            img_w (int): Image width.

    Methods:
            transform_annotations(): Transform dataset annotations to the model's target format.
            filter_lane(lane): Filter duplicate Y-coordinates from a lane.
            transform_annotation(anno, img_wh=None): Transform an annotation to the model's target format.
            sample_lane(points, sample_ys): Sample points along a lane.
            label_to_lanes(label): Convert labels to lane objects.
            draw_annotation(idx, label=None, pred=None, img=None): Visualize annotations and predictions on an image.
            lane_to_linestrings(lanes): Convert lanes to LineString objects.
            linestrings_to_lanes(lines): Convert LineString objects to lanes.
            __getitem__(idx): Retrieve an item (image and label) from the dataset.
            __len__(): Get the length of the dataset.
    """

    def __init__(self,
                 S=72,
                 dataset='culane',
                 augmentations=None,
                 normalize=False,
                 img_size=(320, 800),
                 aug_chance=1.,
                 **kwargs):
        """
        Initialize the LaneDataset with specified parameters.

        Args:
            S (int): Number of strips. Images often have a high resolution along the vertical axis (height), especially in cases
              where the road or lane stretches out ahead. Instead of processing the lane annotations or detecting lanes at each i
              ndividual pixel row, it's computationally more efficient and sometimes more robust to split the image into horizontal
                strips and detect lanes within these strips.
                For instance, consider an image of height H. If S = 72, then each strip's height is H/71 (because n_strips = S - 1).
                The lane detection algorithm will then only need to work with 71 strips as opposed to H individual rows, 
                thus reducing computational complexity.

                Within each strip, the algorithm might:

                    Interpolate or extrapolate lane positions.
                    Determine if a lane is present or not.
                    Decide other lane-related characteristics.

                By working with strips, the algorithm also becomes more robust against noise or small irregularities, since it processes a chunk of pixels together as opposed to individual rows.

                In summary, S (and hence n_strips) allows the algorithm to work with a down-sampled version of the vertical axis of the image, making it both efficient and robust.
            dataset (str): Dataset type ('culane' supported).
            augmentations (list): List of data augmentation configurations.
            normalize (bool): Whether to normalize image data.
            img_size (tuple): Tuple representing the image size (height, width).
            aug_chance (float): Probability of applying augmentations.
            **kwargs: Additional keyword arguments.
        """
        super(LaneDataset, self).__init__()
        # if dataset == 'tusimple':
        #     self.dataset = TuSimple(**kwargs)
        if dataset == 'culane':
            self.dataset = CULane(**kwargs)
        # elif dataset == 'llamas':
        #     self.dataset = LLAMAS(**kwargs)
        # elif dataset == 'nolabel_dataset':
        #     self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()
        self.n_strips = S - 1
        self.n_offsets = S
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.strip_size = self.img_h / self.n_strips

        # y at each x offset
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        else:
            augmentations = []

        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.resizing_coordinates = {}

    @property
    def annotations(self):
        return self.dataset.annotations

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



    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = cv2.imread(item['path'])
        mask = self.dataset.create_mask(item['lanes'])

        # Resize image
        # img = cv2.resize(img, (self.img_w, self.img_h))
        # mask = cv2.resize(mask, (self.img_w, self.img_h))
        # To try to skew away from the underlying distribution, select random 320x800 position
        # If we already tried to resize a frame from this video, resize it based on that
        video_path = item['path'].split('/')[-2]
        if video_path in self.resizing_coordinates:
            y, x = self.resizing_coordinates[video_path]
            img = img[y:y + self.img_h, x:x + self.img_w]
            mask = mask[y:y + self.img_h, x:x + self.img_w]
        else:
            y = np.random.randint(0, img.shape[0] - self.img_h)
            x = np.random.randint(0, img.shape[1] - self.img_w)
            self.resizing_coordinates[video_path] = (y, x)
            img = img[y:y + self.img_h, x:x + self.img_w]
            mask = mask[y:y + self.img_h, x:x + self.img_w]

        # Standardize image
        img = img / 255
        # Normalize image
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return img, mask, idx

    def __len__(self):
        return len(self.dataset)
