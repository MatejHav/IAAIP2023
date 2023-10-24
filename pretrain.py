import torch
import os
import time
import gc
import json
import numpy as np
import cv2

from torch.optim import AdamW
from tqdm.auto import tqdm
from main import get_dataloader
from models.model_collection import *

import models.resnet_autoencoder as resnet_autoencoder
import models.vision_transformer as vit
from models.vit_autoencoder import ViTAutoencoder

model_name = 'resnet_autoencoder'
save_path = "./models/checkpoints/pretrained_vit/"


def training_loop(num_epochs, dataloaders, model, device):
    print("\n" + ''.join(['#'] * 25) + "\n")
    print(f'PERFORMING PRETRAINING. SAVING INTO {save_path}.')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    # SGD
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = lambda pred, tar: torch.sqrt(torch.nn.MSELoss()(pred, tar))

    losses = {
        'train': [],
        'val': []
    }

    # i = 0

    for epoch in range(num_epochs):
        model.train()
        progress_bar_train = tqdm(dataloaders['train'])
        progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: TBD")
        total_loss_train = 0
        for unmasked, batch, targets, _ in progress_bar_train:
            optimizer.zero_grad()
            unmasked = unmasked.to(device)
            batch = batch.to(device)
            prediction = model(batch)
            loss = loss_function(prediction, unmasked)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {loss.item():.4f}")

        losses['train'].append(total_loss_train / len(progress_bar_train))
        # save the self.encoder model
        torch.save(model.state_dict(), f'{save_path}/pretrained_vit_{epoch}.model')


if __name__ == "__main__":
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA RECOGNIZED.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS RECOGNIZED.")
    else:
        device = torch.device("cpu")
        print("NO GPU RECOGNIZED.")

    # Training Parameters
    num_epochs = 2
    batch_size = 8
    culane_dataloader = {
        'train': get_dataloader('train', batch_size, subset=10),
        'val': get_dataloader('val', batch_size),
        'test': get_dataloader('test', batch_size)
    }

    model = ViTAutoencoder()
    training_loop(num_epochs, culane_dataloader, model, device)
