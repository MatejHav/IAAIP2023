import torch
from lane_dataset import LaneDataset
import random
import numpy as np
from tqdm import tqdm
import cv2


def _worker_init_fn_(_):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def get_train_dataloader():
    split = 'train'
    root = './data/'
    train_dataset = LaneDataset(split=split, root=root, load_formatted=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=4,
                                               shuffle=False,  # !
                                               num_workers=1,
                                               worker_init_fn=_worker_init_fn_)

    return train_loader


idx = 0

if __name__ == '__main__':
    train_loader = get_train_dataloader()
    num_epochs = 10
    pbar = tqdm(train_loader)
    for i, (images, lanes, _) in enumerate(pbar):
        # What we're doing here: the original tensor is likely in the format (channels, height, width)
        # commonly used in PyTorch. However, many image processing libraries expect the channels to be
        # the last dimension, i.e., (height, width, channels).
        # .permute(1, 2, 0) swaps the dimensions so that the channels dimension becomes the last one.
        # This is done to match the channel order expected by most image display functions.
        img0 = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        label_img0, _, _ = train_loader.dataset.draw_annotation(i, img0)
        cv2.imshow('img', label_img0)
        cv2.waitKey(0)

        # img1 = (images[1].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # label_img1, _, _ = train_loader.dataset.draw_annotation(i)
        # cv2.imshow('img', label_img1)
        # cv2.waitKey(0)
        #
        # img2 = (images[2].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # label_img2, _, _ = train_loader.dataset.draw_annotation(i)
        # cv2.imshow('img', label_img2)
        # cv2.waitKey(0)
        #
        # img3 = (images[3].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # label_img3, _, _ = train_loader.dataset.draw_annotation(i)
        # cv2.imshow('img', label_img3)
        # cv2.waitKey(0)

        # for batch_idx in range(lanes.shape[0]):
        #     lanes_np = lanes.cpu().numpy()
        #     lane_coords_for_batch = train_loader.dataset.label_to_lanes(lanes_np[batch_idx])
        #     print(lane_coords_for_batch)

        # annotations = train_loader.dataset.annotations[i]['old_anno']
        # print(annotations)
        # exit()
