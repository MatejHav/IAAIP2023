import gc

import torch
from culane.lane_dataset import LaneDataset, IMAGENET_MEAN, IMAGENET_STD
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
from tqdm import tqdm
import cv2

from models.vit_autoencoder import ViTAutoencoder


def _worker_init_fn_(_):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def get_dataloader(split: str = 'train', batch_size: int = 30, subset=30, shuffle=True):
    root = './culane/data/'
    dataset = LaneDataset(split=split, root=root, subset=subset, normalize=True)

    if not shuffle:
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            worker_init_fn=_worker_init_fn_)
        return loader
    batch_sampler = np.arange(0, len(dataset))
    batch_sampler = np.pad(batch_sampler, (0, max(len(dataset) - (len(dataset) // batch_size + 1) * batch_size,
                                                  (len(dataset) // batch_size + 1) * batch_size - len(dataset))),
                           mode='constant', constant_values=0)
    batch_sampler = batch_sampler.reshape((len(dataset) // batch_size + 1, batch_size))
    np.random.shuffle(batch_sampler)
    loader = DataLoader(dataset=dataset,
                        batch_sampler=batch_sampler,
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
    batch_size = 8
    root = './culane/data/'
    dataset = LaneDataset(split='train', root=root, subset=10, normalize=True)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=_worker_init_fn_)
    pbar = tqdm(loader)
    # backbone = Backbone('resnet34')
    # backbone.to(device)
    from models.model_collection import get_vitt
    backbone, model = get_vitt(device)
    state_dict = torch.load('models/checkpoints/vitt/model_1698322627_vitt_9.model')
    model.load_state_dict(state_dict)
    backbone.to(device)
    model.to(device)
    last = None
    sample_size = 100
    truth_color = [(1,0,0), (1,0.3,0), (1,0.6,0), (1,1,0)]
    pred_color = [(0,0,1), (0,0.3,1), (0,0.6,1), (0,1,1)]
    for i, (images, masked_images, masks, idx) in enumerate(pbar):
        # RUNNING TRAINED MODEL PREDICTIONS
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            batch_of_segments = backbone(images)
            labels = model(batch_of_segments)
        labels = labels.cpu()
        masks = masks.cpu()
        if last is None:
            last = labels
        else:
            last = labels

        batch_size = images.shape[0]
        for j, img in enumerate(images):
            y, x = np.where(labels[j] >= 0.5)
            y_gt, x_gt = np.where(masks[j] >= 0.5)
            y_over, x_over = np.where(np.logical_and(labels[j] >= 0.5, masks[j] >= 0.5))
            img = img.cpu().numpy()
            img = np.transpose(img, axes=[1, 2, 0])
            # img = img * IMAGENET_STD + IMAGENET_MEAN
            img[y, x, 0] = labels[j, y, x]
            img[y, x, 1] = 0
            img[y, x, 2] = 0
            img[y_gt, x_gt, 0] = 0
            img[y_gt, x_gt, 1] = 1
            img[y_gt, x_gt, 2] = 0
            img[y_over, x_over, 0] = 0
            img[y_over, x_over, 1] = 0
            img[y_over, x_over, 2] = 1
            cv2.imshow('original', img)
            cv2.waitKey(500)
