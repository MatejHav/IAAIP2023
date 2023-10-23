import cv2
import numpy as np
import imgaug.augmenters as iaa
import torch
from PIL import Image
from imgaug.augmenters import Resize
from timm.models.vision_transformer import PatchEmbed
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset

from culane import config
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
                 img_size=(576, 576), # 320, 800
                #  img_size=(513, 1024),
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

        if "load_formatted" in kwargs:
            self.load_formatted = kwargs['load_formatted']
        else:
            self.load_formatted = True

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

    def transform_annotation(self, anno, img_wh=None):
        """
        Transforms an annotation from its original format to the format 
        that the model expects.
        
        Args:
            anno (dict): Original annotation.
            img_wh (tuple, optional): Width and height of the image. If not provided, 
                                      these will be derived from the annotation.
                                      
        Returns:
            dict: Transformed annotation.
        """
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates between 0 and 1
        old_lanes = [[(x / float(img_w), y / float(img_h)) for x, y in lane]
                     for lane in old_lanes]
        # create transformed annotations
        # 32 samples of y coordinate
        # 80 samples of x coordinate
        # 3 -> length, angle, probability
        lanes = np.zeros((GROUND_TRUTH_GRID[0], GROUND_TRUTH_GRID[1], 3), dtype=np.float32)

        for lane_idx, lane in enumerate(old_lanes):
            # Last point will be added automatically with the length property of lanes
            for index, point in enumerate(lane[:-1]):
                next_point = lane[index + 1]
                vector = (next_point[0] - point[0], next_point[1] - point[1])
                # Length can be max sqrt(0.1**2 + 0.1**2) = 0.141
                length = min(0.141, np.sqrt(vector[0] ** 2 + vector[1] ** 2))
                # Angle: arctan(y/x)
                angle = np.arctan(vector[1] / (vector[0] + 1e-5))
                # Closest bin to the coordinates
                y_bin = min(GROUND_TRUTH_GRID[0] - 1, int(point[1] * (GROUND_TRUTH_GRID[0] - 1)))
                x_bin = min(GROUND_TRUTH_GRID[1] - 1, int(point[0] * (GROUND_TRUTH_GRID[1] - 1)))
                lanes[y_bin, x_bin, 0] = length
                lanes[y_bin, x_bin, 1] = angle
                # Probability of ground truth is always 1
                lanes[y_bin, x_bin, 2] = 1

        return {'path': anno['path'], 'lanes': lanes}

    def label_to_lanes(self, label):
        """
        Convert labels to lane objects.

        Args:
            label (np.ndarray): Model output labels.

        Returns:
            list: List of lane objects.
        """
        img_height = self.img_h
        img_width = self.img_w
        lanes = []
        for y in range(GROUND_TRUTH_GRID[0]):
            for x in range(GROUND_TRUTH_GRID[1]):
                if label[y, x, 2] >= 0.5:
                    length = label[y, x, 0]
                    angle = label[y, x, 1]
                    temp = [(int(x / GROUND_TRUTH_GRID[1] * img_width), int(y / GROUND_TRUTH_GRID[0] * img_height)),
                            (int((x / GROUND_TRUTH_GRID[1] + np.cos(angle) * length) * img_width),
                             int((y / GROUND_TRUTH_GRID[0] + np.sin(angle) * length) * img_height))]
                    lanes.append(temp)
        return lanes

    def draw_annotation(self, idx, label=None, pred=None, img=None):
        """
        Visualize annotations and predictions on an image.

        Args:
            idx (int): Index of the dataset item.
            label (np.ndarray): Ground truth labels.
            pred (np.ndarray): Predicted labels.
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Image with visualized annotations and predictions.
            int: False positives.
            int: False negatives.
        """
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, _, _ = self.__getitem__(idx)
            if label is None:
                img, label, _ = self[idx]
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            if label is None:
                _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
        img = cv2.resize(img, (self.img_w, self.img_h))

        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        for matches, accs, datum in data:
            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
                    color = PRED_MISS_COLOR
                start_point = l[0]
                end_point = l[1]
                img = cv2.line(img, start_point, end_point,
                               color=color,
                               thickness=3 if matches is None else 3)
        return img, fp, fn

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

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """  # (128, 256, 3)
        x = torch.tensor(x)
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
        x1 = x
        x = x.float()
        self.patch_embed = PatchEmbed((576, 576), 16, 3, 768)
        # embed patches
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768), requires_grad=False)
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        # mask = mask.unsqueeze(dim=0)
        mask = self.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        x1 = torch.einsum('nchw->nhwc', x1)
        # masked image
        im_masked = x1 * (1 - mask)


        return im_masked, mask, ids_restore
    def __getitem__(self, idx):
        item = self.dataset[idx]

        img = cv2.imread(item['path'])
        # For some reason cv2 resize wants to have width and then height
        original_shape = (img.shape[1], img.shape[0])

        # Resize image
        # img = cv2.resize(img, (self.img_w, self.img_h))
        # To try to skew away from the underlying distribution, select random 320x800 position
        # If we already tried to resize a frame from this video, resize it based on that
        video_path = item['path'].split('/')[-2]
        if video_path in self.resizing_coordinates:
            y, x = self.resizing_coordinates[video_path]
            img = img[y:y+self.img_h, x:x+self.img_w]
        else:
            y = np.random.randint(0, img.shape[0] - self.img_h)
            x = np.random.randint(0, img.shape[1] - self.img_w)
            self.resizing_coordinates[video_path] = (y, x)
            img = img[y:y + self.img_h, x:x + self.img_w]

        # Standardize image
        img = img / 255
        img = np.array(img)
        img, mask2, ids_restore = self.random_masking(img, mask_ratio=config.mask_ratio)
        img = img.squeeze(dim=0)
        img = np.array(img)
        # Normalize image
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        if not self.load_formatted:
            transformed = self.transform_annotation(item)['lanes']
            # mask = self.create_mask(transformed)
            mask = cv2.resize(self.create_mask(transformed), original_shape)[y:y+self.img_h, x:x + self.img_w]
            return img, transformed, mask, idx
        # mask = self.create_mask(item['lanes'])
        mask = cv2.resize(self.create_mask(item['lanes']), original_shape)[y:y + self.img_h, x:x + self.img_w]
        return img, item['lanes'], mask, idx

    def __len__(self):
        return len(self.dataset)
