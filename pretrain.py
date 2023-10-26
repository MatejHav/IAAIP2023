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
import models.vision_transformer_with_pytorch as PyTorchVisionTransformer

model_name = 'resnet_autoencoder'
save_path = "./models/checkpoints/mask/"

SHWO_IMAE_PROGRESS = False
MODULO = 100


def training_loop(num_epochs, dataloaders, model, device):
    print("\n" + ''.join(['#'] * 25) + "\n")
    print(f'PERFORMING PRETRAINING. SAVING INTO {save_path}.')
    if not os.path.exists(save_path):
        os.mkdir(model_name["path"])

    saved_time = int(time.time())
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()

    losses = {
        'train': [],
        'val': []
    }

    i = 0

    for epoch in range(num_epochs):
        model.train()
        progress_bar_train = tqdm(dataloaders['train'])
        progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: TBD")
        total_loss_train = 0
        for batch, targets, masks, _ in progress_bar_train:

            batch = batch.to(device)
            prediction = model(batch)
            loss = loss_function(prediction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {total_loss_train / len(progress_bar_train):.4f}")

            if SHWO_IMAE_PROGRESS == True:
                if i % MODULO == 0: # and epoch > 2:
                    output_image = prediction[0].cpu().detach().numpy()
                    output_image = np.transpose(output_image, (1, 2, 0))

                    # Scale the image to the range [0, 255] if it's not already in that range.
                    if output_image.max() <= 1:
                        output_image = (output_image * 255).astype(np.uint8)
                    else:
                        output_image = output_image.astype(np.uint8)

                    # Save without the BGR to RGB conversion, assuming the model output is already in RGB format.
                    cv2.imwrite(f'./models/checkpoints/outputs/{i}.jpg', output_image)
                    cv2.imshow('img', output_image)
                    cv2.waitKey(1)

        i = i + 1

        losses['train'].append(total_loss_train / len(progress_bar_train))

    # save model
    torch.save(model, f'{save_path}/{saved_time}_{epoch}.model')

    # save loss to json file
    with open(f'{save_path}/model_{saved_time}_{epoch}.json', 'w') as file:
        json.dump(losses, file)


# def testing_loop(num_epochs, dataloaders, model, device):
#     print("\n" + ''.join(['#'] * 25) + "\n")
#     print(f'PERFORMING PRETRAINING | TESTING MODE. SAVING INTO {save_path}.')
#     if not os.path.exists(save_path):
#         os.mkdir(model_name["path"])

#     saved_time = int(time.time())
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr=0.001)
#     # SGD
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#     loss_function = torch.nn.MSELoss()

#     losses = {
#         'train': [],
#         'val': []
#     }

#     i = 0

#     for epoch in range(num_epochs):
#         model.eval()
#         progress_bar_train = tqdm(dataloaders['test'])
#         progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: TBD")
#         total_loss_train = 0
#         for batch, targets, masks, _ in progress_bar_train:

#             batch = batch.to(device)
#             prediction = model(batch)
#             loss = loss_function(prediction, batch)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss_train += loss.item()
#             progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {total_loss_train / len(progress_bar_train):.4f}")

#             if SHWO_IMAE_PROGRESS == True:
#                 if i % MODULO == 0:
#                     output_image = prediction[0].cpu().detach().numpy()
#                     output_image = np.transpose(output_image, (1, 2, 0))

#                     # Scale the image to the range [0, 255] if it's not already in that range.
#                     if output_image.max() <= 1:
#                         output_image = (output_image * 255).astype(np.uint8)
#                     else:
#                         output_image = output_image.astype(np.uint8)

#                     input_image = batch[0].cpu().detach().numpy()
#                     input_image = np.transpose(input_image, (1, 2, 0))
#                     # Save without the BGR to RGB conversion, assuming the model output is already in RGB format.
#                     # cv2.imwrite(f'./models/checkpoints/outputs/{i}.jpg', output_image)
#                     cv2.imshow('input', input_image)
#                     cv2.imshow('output', output_image)
#                     cv2.waitKey(1)


#         i = i + 1

#         losses['train'].append(total_loss_train / len(progress_bar_train))

#     # save model
#     torch.save(model, f'{save_path}/model_{saved_time}_{epoch}.model')

#     # save loss to json file
#     with open(f'{save_path}/model_{saved_time}_{epoch}.json', 'w') as file:
#         json.dump(losses, file)

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
    num_epochs = 100
    batch_size = 5
    culane_dataloader = {
        'train': get_dataloader('train', batch_size, subset=100),
        'val': get_dataloader('val', batch_size),
        'test': get_dataloader('test', batch_size)
    }

    # model = resnet_autoencoder.ResNetAutoencoder()
    # model = vit.VisionTransformer()

    model = PyTorchVisionTransformer.ViTAutoencoder()

    training_loop(num_epochs, culane_dataloader, model, device)
    # testing_loop(num_epochs, culane_dataloader, model, device)

    # save the self.encoder model
    torch.save(model.vit, f'{save_path}/encoder.model')
    torch.save(model.decoder, f'{save_path}/decoder.model')
    torch.save(model.state_dict(), f'{save_path}/VitAutoEncoder.model')