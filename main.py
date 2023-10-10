import gc

import torch
from culane.lane_dataset import LaneDataset
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import cv2

from models.backbone.backbone import Backbone


def _worker_init_fn_(_):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def get_dataloader(split: str = 'train', batch_size: int = 30, subset=100):
    root = './culane/data/'
    dataset = LaneDataset(split=split, root=root, load_formatted=False, subset=subset, normalize=False)
    loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=False,  # Should shuffle the batches for each epoch
                              worker_init_fn=_worker_init_fn_)
    return loader


idx = 0

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA RECOGNIZED.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS RECOGNIZED.")
    else:
        device = torch.device("cpu")
        print("NO GPU RECOGNIZED.")
    train_loader = get_dataloader('train', batch_size=30, subset=10)
    num_epochs = 10
    pbar = tqdm(train_loader)
    backbone = Backbone('grid')
    backbone.to(device)
    model = torch.load('./models/checkpoints/mask/model_1696942501_0.model')
    model.to(device)
    for i, (images, lanes, masks, _) in enumerate(pbar):
        # What we're doing here: the original tensor is likely in the format (channels, height, width)
        # commonly used in PyTorch. However, many image processing libraries expect the channels to be
        # the last dimension, i.e., (height, width, channels).
        # .permute(1, 2, 0) swaps the dimensions so that the channels dimension becomes the last one.
        # This is done to match the channel order expected by most image display functions.
        # batch_size = images.shape[0]
        # for j, _ in enumerate(images):
        #     # Need to multiply the batch_size with the index to get the actual correct frame
        #     label_img, _, _ = train_loader.dataset.draw_annotation(batch_size * i + j)
        #     cv2.imshow('img', label_img)
        #     cv2.waitKey(500)
        # cv2.waitKey(0)

        # RUNNING TRAINED MODEL PREDICTIONS
        images = images.to(device)
        with torch.no_grad():
            batch_of_segments = backbone(images)
            labels = model(batch_of_segments)
        labels = labels.cpu()
        del batch_of_segments
        gc.collect()
        torch.cuda.empty_cache()
        batch_size = images.shape[0]
        for j, img in enumerate(images):
            # Need to multiply the batch_size with the index to get the actual correct frame
            # img = np.swapaxes(img, axis1=0, axis2=1)
            # img = np.swapaxes(img, axis1=1, axis2=2)
            y, x = np.where(labels[j] >= 0.3)
            img = img.cpu().numpy()
            img[0, y, x] = labels[j, y, x]
            img[1, y, x] = 1
            img[2, y, x] = 1
            img = np.transpose(img, axes=[1, 2, 0])
            cv2.imshow('img', img)
            cv2.waitKey(50)
        # cv2.waitKey(0)
