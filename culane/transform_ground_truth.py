import torch

from lane_dataset import LaneDataset
from main import get_train_dataloader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    root = './data/'
    dataset = LaneDataset(split='train', root=root, load_formatted=False)
    bar = tqdm(dataset)
    for i, (images, lanes, _) in enumerate(bar):
        item = dataset.dataset[i]
        torch.save(lanes, item['path'][:-4] + '_lines_formatted.txt')
        bar.set_description(f'Video: {item["path"].split("/")[-3]}/{item["path"].split("/")[-2]}')
