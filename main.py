import gc

import torch
from culane.lane_dataset import LaneDataset, IMAGENET_MEAN, IMAGENET_STD
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import cv2



def _worker_init_fn_(_):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2 ** 32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


def get_dataloader(split: str = 'train', batch_size: int = 30, subset=30, load_formatted=False, load_fit=False, save_fit=False, save_formatted=False):
    root = './culane/data/'
    dataset = LaneDataset(split=split, root=root, save_fit=save_fit, save_formatted=save_formatted, load_formatted=load_formatted, load_fit=load_fit, subset=subset, normalize=True)
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
    batch_size = 15
    root = './culane/data/'
    dataset = LaneDataset(split='train', root=root, load_formatted=False, load_fit=False, save_fit=True, subset=10,
                          normalize=True)
    batch_sampler = np.arange(0, len(dataset))
    batch_sampler = np.pad(batch_sampler, (0, max(len(dataset) - (len(dataset) // batch_size + 1) * batch_size,
                                                  (len(dataset) // batch_size + 1) * batch_size - len(dataset))),
                           mode='constant', constant_values=0)
    batch_sampler = batch_sampler.reshape((len(dataset) // batch_size + 1, batch_size))
    np.random.shuffle(batch_sampler)
    loader = DataLoader(dataset=dataset,
                        batch_sampler=batch_sampler,
                        worker_init_fn=_worker_init_fn_)
    pbar = tqdm(loader)
    # backbone = Backbone('resnet34')
    # backbone.to(device)
    from models.model_collection import get_vitt
    backbone, model = get_vitt(device)
    state_dict = torch.load('./models/checkpoints/vitt/model_1697541206_vitt_0.model')
    model.load_state_dict(state_dict)
    model.to(device)
    last = None
    sample_size = 100
    truth_color = [(1,0,0), (1,0.3,0), (1,0.6,0), (1,1,0)]
    pred_color = [(0,0,1), (0,0.3,1), (0,0.6,1), (0,1,1)]

    # for i, (images, lanes, masks, idx) in enumerate(pbar):
        # images = images.to(device)
        # with torch.no_grad():
        #     predictions = model(images)
        # print((lanes - predictions.cpu()).mean(dim=0), torch.nn.MSELoss()(predictions.cpu(), lanes))
        # for j in range(len(images)):
        #     img = images[j].cpu().numpy()
        #     img = np.transpose(img, [1, 2, 0])
        #     img = img * IMAGENET_STD + IMAGENET_MEAN
        #
        #     lane = lanes[j].cpu().numpy()
        #     for k in range(len(lane)):
        #         x = np.linspace(lane[k][-2], lane[k][-1], sample_size)
        #         y = np.array(
        #             list(map(lambda x: lane[k][0] * x ** 2 + lane[k][1] * x + lane[k][2], x)))
        #         # Set back to image coordinates
        #         x = x * len(img[0])
        #         y = y * len(img)
        #
        #         for p in range(1, sample_size):
        #             if x[p - 1] < 0 or x[p - 1] >= len(img[0]) or y[p - 1] < 0 or y[p - 1] >= len(img):
        #                 continue
        #             img = cv2.line(img.copy(), (int(x[p - 1]), int(y[p - 1])), (int(x[p]), int(y[p])),
        #                            color=truth_color[k],
        #                            thickness=3)
        #
        #     lane = predictions[j].cpu().numpy()
        #     for k in range(len(lane)):
        #         x = np.linspace(lane[k][-2], lane[k][-1], sample_size)
        #         y = np.array(
        #             list(map(lambda x: lane[k][0] * x ** 2 + lane[k][1] * x + lane[k][2], x)))
        #         # Set back to image coordinates
        #         x = x * len(img[0])
        #         y = y * len(img)
        #
        #         for p in range(1, sample_size):
        #             if x[p-1] < 0 or x[p-1] >= len(img[0]) or y[p-1] < 0 or y[p-1] >= len(img):
        #                 continue
        #             img = cv2.line(img.copy(), (int(x[p - 1]), int(y[p - 1])), (int(x[p]), int(y[p])), color=pred_color[k], thickness=3)
        #     cv2.imshow('img', img)
        #     cv2.waitKey(50)

        # RUNNING TRAINED MODEL PREDICTIONS
        # images = images.to(device)
        # masks = masks.to(device)
        # with torch.no_grad():
        #     batch_of_segments = backbone(images)
        #     labels = model(batch_of_segments)
        # labels = labels.cpu()
        # masks = masks.cpu()
        # if last is None:
        #     last = labels
        # else:
        #     print((last - labels).std())
        #     last = labels
        #
        # batch_size = images.shape[0]
        # for j, img in enumerate(images):
        #     y, x = np.where(labels[j] >= 0.5)
        #     y_gt, x_gt = np.where(masks[j] >= 0.5)
        #     img = img.cpu().numpy()
        #     img = np.transpose(img, axes=[1, 2, 0])
        #     img = img * IMAGENET_STD + IMAGENET_MEAN
        #     img[y, x, 0] = labels[j, y, x]
        #     img[y, x, 1] = 0
        #     img[y, x, 2] = 0
        #     img[y_gt, x_gt, 0] = 0
        #     img[y_gt, x_gt, 1] = 1
        #     img[y_gt, x_gt, 2] = 0
        #     cv2.imshow('img', img)
        #     cv2.waitKey(50)
