import os
import time
import json

import torch.nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from main import *
from models.model_collection import *


def iou(pred: torch.Tensor, tar: torch.Tensor, threshold=0.5):
    if pred.min() < 0 or pred.max() > 1:
        pred = nn.Sigmoid()(pred)
    up = torch.logical_and(pred >= threshold, tar >= threshold).sum(dim=[1, 2])
    down = torch.logical_or(pred >= threshold, tar >= threshold).sum(dim=[1, 2])
    return ((up + 1e-6) / (down + 1e-6)).mean()

def iou_loss(pred: torch.Tensor, tar: torch.Tensor):
    up = (pred * tar).sum(dim=[1, 2])
    down = (0.5 * pred + 0.5 * tar).sum(dim=[1, 2])
    return ((up + 1e-6) / (down + 1e-6)).mean()

def training_loop(num_epochs, dataloaders, models, device):
    for model_name in models:
        print("\n" + ''.join(['#'] * 25) + "\n")
        print(f'TRAINING MODEL {model_name}. SAVING INTO {models[model_name]["path"]}.')
        if not os.path.exists(models[model_name]['path']):
            os.mkdir(models[model_name]['path'])
        saved_time = int(time.time())
        model = models[model_name]['model']
        model.to(device)
        optimizer = AdamW(model.parameters(), weight_decay=1e-7, lr=1e-5)
        loss_function = lambda pred, tar : 1 - iou_loss(pred, tar)
        for epoch in range(num_epochs):
            losses = {
                'train': [],
                'val': []
            }
            ious = {
                'train': [],
                'val': []
            }
            # Setup progress bars
            dataloader = dataloaders['train'][0](*dataloaders['train'][1])
            progress_bar_train = tqdm(dataloader)
            progress_bar_train.set_description(f"[STARTING TRAINING] | ")
            model.training = True
            for batch, targets in progress_bar_train:
                optimizer.zero_grad()
                # Load batch into memory
                batch = batch.to(device)
                targets = targets.to(device)
                # Make predictions
                predictions = model(batch, targets)
                # Compute loss
                loss = loss_function(predictions, targets)
                loss.backward()
                # Learn
                optimizer.step()
                # Save loss for printouts
                intersect_over_union = iou(predictions, targets).item()
                losses['train'].append(loss.item())
                ious['train'].append(intersect_over_union)
                progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {loss.item():.3f} |"
                                                   f" MEAN LOSS: {np.mean(losses['train']):.3f} |"
                                                   f" MEAN IOU: {np.mean(ious['train']):.3f} |")

            # Validate the model
            dataloader = dataloaders['val'][0](*dataloaders['val'][1])
            progress_bar_val = tqdm(dataloader)
            progress_bar_val.set_description(f"[VALIDATION] | EPOCH {epoch} | ")
            total_loss_val = 0
            model.training = False
            for batch, targets in progress_bar_val:
                batch = batch.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    predictions = model(batch, targets)
                    loss = loss_function(predictions, targets)
                    losses['val'].append(loss.item())
                    intersect_over_union = iou(predictions, targets).item()
                    ious['val'].append(intersect_over_union)
            print(f'EPOCH {epoch} | MEAN VALIDATION LOSS: {np.mean(losses["val"])} | MEAN VALIDATION IOU: {np.mean(ious["val"])}')

            # Save model and statistics
            path = os.path.join(models[model_name]['path'], f"model_{saved_time}_{model_name}_{epoch}.model")
            torch.save(model.state_dict(), path)
            os.makedirs(os.path.join(models[model_name]['path'], 'stats'), exist_ok=True)
            with open(os.path.join(models[model_name]['path'], 'stats',
                                   f"loss_model_{saved_time}_{model_name}_{epoch}.json"),
                      'w') as file:
                json.dump(losses, file)
            with open(os.path.join(models[model_name]['path'], 'stats',
                                   f"iou_model_{saved_time}_{model_name}_{epoch}.json"),
                      'w') as file:
                json.dump(ious, file)
            print(f"MODEL SAVED IN {path}")


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
    num_epochs = 200
    batch_size = 16
    culane_dataloader = {
        'train': (get_dataloader, ('train', batch_size, 100, True)),
        'val': (get_dataloader, ('val', batch_size, 100, True)),
        'test': (get_dataloader, ('test', batch_size, 100, True))
    }
    models = {
        "vitt": {"model": get_vitt(device), "path": "./models/checkpoints/vitt/"}
    }

    training_loop(num_epochs, culane_dataloader, models, device)
