import os

import torch
import models.vision_transformer_with_pytorch as PyTorchVisionTransformer
from torch.optim import AdamW
from tqdm.auto import tqdm
import cv2
import json
import numpy as np
import time
from main import get_dataloader
from MAE.masked_autoencoder import MaskedAutoencoderViT
from mae_original_encoder import OriginalEncoder
from mae_original_decoder import OriginalDecoder

model_name = 'resnet_autoencoder'
save_path = "../models/checkpoints/mask/"

SHWO_IMAE_PROGRESS = True
MODULO = 50
PATCH_SIZE = 16
MASKING_RATIO = 0.75
IMAGE_SIZE = 576



def training_loop_mae(num_epochs, dataloaders, mae, device):
    print("\n" + ''.join(['#'] * 25) + "\n")
    print(f'PERFORMING PRETRAINING. SAVING INTO {save_path}.')
    if not os.path.exists(save_path):
        os.mkdir(save_path + model_name)

    saved_time = int(time.time())
    mae.to(device)
    optimizer = AdamW(mae.parameters(), lr=0.001)

    losses = {
        'train': [],
        'val': []
    }

    i = 0

    for epoch in range(num_epochs):
        mae.train()
        progress_bar_train = tqdm(dataloaders['train'])
        progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: TBD")
        total_loss_train = 0
        for batch, _, _, _ in progress_bar_train:
            batch = batch.to(device)
            loss, prediction, mask = mae.forward(batch, MASKING_RATIO)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
            progress_bar_train.set_description(
                f"[TRAINING] | EPOCH {epoch} | LOSS: {total_loss_train / len(progress_bar_train):.4f}")

            if SHWO_IMAE_PROGRESS == True:
                if i % MODULO == 0 and epoch > 2:
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
    torch.save(mae, f'{save_path}/model_{saved_time}_{epoch}.model')

    # save loss to json file
    with open(f'{save_path}/model_{saved_time}_{epoch}.json', 'w') as file:
        json.dump(losses, file)



def main():
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

    num_epochs = 20
    batch_size = 2
    culane_dataloader = {
        'train': get_dataloader('train', batch_size, subset=100),
        'val': get_dataloader('val', batch_size),
        'test': get_dataloader('test', batch_size)
    }

    # vision_transformer = PyTorchVisionTransformer.ViTAutoencoder()
    # encoder = vision_transformer.vit
    # decoder = vision_transformer.decoder

    decoder = OriginalDecoder()
    encoder = OriginalEncoder()

    mae = MaskedAutoencoderViT(encoder, decoder, img_size=IMAGE_SIZE)

    training_loop_mae(num_epochs, culane_dataloader, mae, device)


if __name__ == "__main__":
    main()