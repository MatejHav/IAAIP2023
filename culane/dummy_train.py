import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import numpy as np
import dataloader as ds
import torchvision 
import torchvision.transforms


if __name__ == '__main__':
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
        ds.CULaneDataset(dataset_path='/home/charles/Desktop/IAAIP-Transformer/culane/data/list', data_list='train_gt', transform=None),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)
    

    # ## Example of doing random scaling and normalizing into the data before feeding to model: 
    # train_loader_with_random_scaling = torch.utils.data.DataLoader(
    #     ds.CULaneDataset(dataset_path='/Users/charlesdowns/Desktop/iaaip-data-loaders/culane/data/list', data_list='train_gt', transform=torchvision.transforms.Compose([
    #         tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
    #         tf.GroupRandomCropRatio(size=(288, 800)),
    #         tf.GroupNormalize(mean=(0.3598, 0.3653, 0.3662), std=(0.2573, 0.2663, 0.2756)),
    #     ])),
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=1, pin_memory=True)
    

    # val loader can be done in a similar way as above, with tf.Bla as intermediate processing 
    """
        val_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
            tf.GroupRandomCropRatio(size=(args.img_width, args.img_height)),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)
    """

    # Main dataloader training loop - pretty standard/default. 
    for i, (input, target, exist) in enumerate(train_loader):
        print(i, input.shape, target.shape, exist.shape)

        #TODO: use tqdm to give a proper progress bar
        # URGENT! Don't do batch size loop here! this is implicit with dataloader?
        for i in range(batch_size): 
            # img = input image, label = ground truth, binary_mask = ground truth mask, exist = whether lanes exist
            img = input[i].numpy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # preprocessing on label/mask to actually make it visible... (it wasn't by default....)
            label = target[i].numpy()
            # convert to binary mask (should already be, but just in case)
            binary_mask = (label > 0).astype(np.uint8) 
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
            # streth values to [0, 255] range 
            binary_mask = binary_mask * 255

            # reduce opacity of the mask and overlay it on the image
            img = cv2.addWeighted(img, 1, binary_mask, 0.5, 0)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # print whether lanes exist
            print("existing lines are: (from left to right)")
            print(exist[i].numpy())