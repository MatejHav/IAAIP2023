import torch

from culane.lane_dataset import LaneDataset
from tqdm import tqdm

if __name__ == "__main__":
    root = './culane/data/'
    dataset = LaneDataset(split='train', root=root, load_formatted=False, subset=100)
    bar = tqdm(dataset)
    for i, (images, lanes, mask, _) in enumerate(bar):
        item = dataset.dataset[i]
        torch.save(lanes, item['path'][:-4] + '_lines_formatted.txt')
        torch.save(mask, item['path'][:-4] + '_mask.txt')
        bar.set_description(f'Video: {item["path"].split("/")[-3]}/{item["path"].split("/")[-2]}')

    root = './culane/data/'
    dataset = LaneDataset(split='val', root=root, load_formatted=False)
    bar = tqdm(dataset)
    for i, (images, lanes, mask, _) in enumerate(bar):
        item = dataset.dataset[i]
        torch.save(lanes, item['path'][:-4] + '_lines_formatted.txt')
        torch.save(mask, item['path'][:-4] + '_mask.txt')
        bar.set_description(f'Video: {item["path"].split("/")[-3]}/{item["path"].split("/")[-2]}')

    root = './culane/data/'
    dataset = LaneDataset(split='test', root=root, load_formatted=False)
    bar = tqdm(dataset)
    for i, (images, lanes, mask, _) in enumerate(bar):
        item = dataset.dataset[i]
        torch.save(lanes, item['path'][:-4] + '_lines_formatted.txt')
        torch.save(mask, item['path'][:-4] + '_mask.txt')
        bar.set_description(f'Video: {item["path"].split("/")[-3]}/{item["path"].split("/")[-2]}')
