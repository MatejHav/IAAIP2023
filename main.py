import gc

import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms import RandomAffine
import matplotlib.pyplot as plt

from culane.lane_dataset import LaneDataset
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import cv2
import imgaug.augmenters as iaa
from scipy import ndimage
import torchvision.transforms.functional as TF

from models.backbone.backbone import Backbone


def _worker_init_fn_(_):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def rotate_mask(mask, angle):
    # Scale float values to the range 0-255
    scaled_mask = np.array(mask, dtype=np.uint8) * 255
    # Create a 3D numpy array with the same shape as image_mask but with 3 channels
    three_channel_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    # Fill all channels with the scaled mask
    for i in range(3):
        three_channel_mask[:, :, i] = scaled_mask

    augm = iaa.Affine(rotate=angle)
    mask = augm.augment_image(three_channel_mask)

    return mask


def print_mask_on_image(img, mask):
    """
    Overlays an image from dataset.getitem() and its mask generated from the groundtruth. Used for testing purposes.
    """
    image_np = img.numpy()

    # Scale the values to the range [0, 255]
    image_np_scaled = image_np * 255.0

    # Clip the values to ensure they are within the valid range
    image_np_clipped = np.clip(image_np_scaled, 0, 255)

    # Convert the numpy array to the uint8 data type
    image_uint8 = torch.tensor(image_np_clipped.astype(np.uint8))

    binary_mask = np.where(mask > 0.5, 1, 0).astype(bool)
    masks = torch.tensor(np.expand_dims(binary_mask, axis=0), dtype=torch.bool)
    overlayed_image = draw_segmentation_masks(image_uint8, masks)
    img_to_print = overlayed_image.numpy()
    img_to_print = np.transpose(img_to_print, (1, 2, 0))

    plt.imshow(img_to_print)
    plt.axis('off')
    plt.show()


def get_dataloader(split: str = 'train', batch_size: int = 30, subset=30):
    root = './culane/data/'
    rotation_angle = random.randint(-45, 45)
    augmentations = [
        {'name': 'HorizontalFlip', 'parameters': {'p': 0.5}},
        {'name': 'Affine', 'parameters': {'rotate': rotation_angle}}
    ]
    dataset = LaneDataset(split=split, root=root, load_formatted=False, subset=subset, normalize=False,
                          augmentations=augmentations)
    it = dataset.__getitem__(44)
    img = it[0]
    mask = it[2]

    print_mask_on_image(img, mask)

    # loader = DataLoader(dataset=dataset,
    #                           batch_size=batch_size,
    #                           shuffle=False,  # Should shuffle the batches for each epoch
    #                           worker_init_fn=_worker_init_fn_)
    # return loader


idx = 0





if __name__ == '__main__':

    get_dataloader()


    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("CUDA RECOGNIZED.")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("MPS RECOGNIZED.")
    # else:
    #     device = torch.device("cpu")
    #     print("NO GPU RECOGNIZED.")
    # train_loader = get_dataloader('train', batch_size=10, subset=30)
    # num_epochs = 10
    # pbar = tqdm(train_loader)
    # backbone = Backbone('resnet34')
    # backbone.to(device)
    # model = torch.load('./models/checkpoints/mask/model_1697034426_0.model')
    # model.to(device)
    # for i, (images, lanes, masks, _) in enumerate(pbar):
    #     # What we're doing here: the original tensor is likely in the format (channels, height, width)
    #     # commonly used in PyTorch. However, many image processing libraries expect the channels to be
    #     # the last dimension, i.e., (height, width, channels).
    #     # .permute(1, 2, 0) swaps the dimensions so that the channels dimension becomes the last one.
    #     # This is done to match the channel order expected by most image display functions.
    #     # batch_size = images.shape[0]
    #     # for j, _ in enumerate(images):
    #     #     # Need to multiply the batch_size with the index to get the actual correct frame
    #     #     label_img, _, _ = train_loader.dataset.draw_annotation(batch_size * i + j)
    #     #     cv2.imshow('img', label_img)
    #     #     cv2.waitKey(500)
    #     # cv2.waitKey(0)
    #
    #     # RUNNING TRAINED MODEL PREDICTIONS
    #     images = images.to(device)
    #     with torch.no_grad():
    #         batch_of_segments = backbone(images)
    #         labels = model(batch_of_segments)
    #     labels = labels.cpu()
    #
    #     del batch_of_segments
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     batch_size = images.shape[0]
    #     for j, img in enumerate(images):
    #         # Need to multiply the batch_size with the index to get the actual correct frame
    #         # img = np.swapaxes(img, axis1=0, axis2=1)
    #         # img = np.swapaxes(img, axis1=1, axis2=2)
    #         # labels_mean = torch.mean(labels[j], dim=0)
    #         # labels_stddev = torch.std(labels[j], dim=0)
    #         # print(labels_mean)
    #         # print(labels_stddev)
    #         # print(labels)
    #         y, x = np.where(labels[j] >= 0.3)
    #         # print('x = ', x, 'y = ', y)
    #         img = img.cpu().numpy()
    #         img[0, y, x] = labels[j, y, x]
    #         img[1, y, x] = 1
    #         img[2, y, x] = 1
    #         img = np.transpose(img, axes=[1, 2, 0])
    #         cv2.imshow('img', img)
    #         cv2.waitKey(50)
    #     # cv2.waitKey(0)
