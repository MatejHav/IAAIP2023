import torch
from lane_dataset import LaneDataset
import random
import numpy as np
import cv2
from tqdm import tqdm, trange
import backbone

#TODO: add const hyperprammns here (instead of hardcoded below)

def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

def get_train_dataloader():
        split = 'train'
        root = '/Users/charlesdowns/Documents/GitHub/IAAIP2023/culane/data'
        train_dataset = LaneDataset(split=split, root=root)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=8,
                                                   shuffle=True,
                                                   num_workers=1,
                                                   worker_init_fn=_worker_init_fn_)
        
        return train_loader
    

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print ("MPS device found.")
# else cuda
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print ("CUDA device found.")
else:
    device = torch.device("cpu")
    print ("No device found, using CPU.")

backbone = backbone.ResNet18Backbone().to(device)
    
if __name__ == '__main__':
    train_loader = get_train_dataloader()
    num_epochs = 10
    
    for epoch in trange(num_epochs):
        backbone.train()
        pbar = tqdm(train_loader)
        for i, (images, lanes, _) in enumerate(pbar):

            #NOTE: move to device before doing anything. 
            
            print(images.shape, lanes.shape)
            
            # some_lane = lanes[0].cpu().numpy()
            # some_lane = train_loader.dataset.label_to_lanes(some_lane)
            # print(some_lane)
            some_lane = train_loader.dataset.label_to_lanes(lanes[i].cpu())
            print(some_lane)

            exit()

            # img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # # fp = false positive, fn = false negative
            # img, fp, fn = train_loader.dataset.draw_annotation(i, img=img)
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            images = images.to(device)
            lanes = lanes.to(device)
            outputs = backbone(images)
            print(outputs.shape)

            # etc...