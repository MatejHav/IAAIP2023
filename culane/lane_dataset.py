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
                 img_size=(576, 576),
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
        self.patch_embed = PatchEmbed((self.img_h, self.img_w), 16, 3, 768)
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
        original = img.copy()
        img = np.array(img)
        img, mask2, ids_restore = self.random_masking(img, mask_ratio=0.0)
        img = img.squeeze(dim=0)
        img = np.array(img)
        # Normalize image
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        original = self.to_tensor(original.astype(np.float32))
        return original, img, mask, idx

    def __len__(self):
        return len(self.dataset)
