import torch
import os
import time
import gc
import json
import numpy as np

from torch.optim import AdamW
from tqdm.auto import tqdm
from main import get_dataloader
from models.model_collection import *


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    In this loss function we compute the MSE loss across all the predictions.
    However, to avoid the network to just output no lanes at all,
    as that is the easier option with way more many empty cells than lane cells,
    we weight the different outputs such that the network has the same weight on lane cells and empty cells.
    :param predictions: Predictions of the network.
    :param targets: Ground truth.plzoveliky
    :return: Total loss
    """
    selection_empty = targets[:, :, :, 2] <= 0.5
    should_be_empty = (predictions[selection_empty] - targets[selection_empty]) ** 2 / torch.sum(selection_empty)
    temp = 0.75 * should_be_empty.sum()
    selection_lane = targets[:, :, :, 2] > 0.5
    should_be_lane = (predictions[selection_lane] - targets[selection_lane]) ** 2 / torch.sum(selection_lane)
    return temp + 0.25 * should_be_lane.sum()


def iou(predictions, targets):
    """
    Computes 1 - IoU to act as a loss function.

    :param predictions: segmentation mask, must be of the same shape as target containing values [0, 1]
    :param targets: binary segmentation mask of the ground truth
    :return: average IoU ove the elements in the batch
    """
    return (1 - (predictions * targets).sum(dim=[1, 2]) / torch.clamp((predictions + targets + 1e-5), min=0, max=1).sum(
        dim=[1, 2])).mean()


def training_loop(num_epochs, dataloaders, models, device):
    for model_name in models:
        print("\n" + ''.join(['#'] * 25) + "\n")
        print(f'TRAINING MODEL {model_name}. SAVING INTO {models[model_name]["path"]}.')
        if not os.path.exists(models[model_name]['path']):
            os.mkdir(models[model_name]['path'])
        saved_time = int(time.time())
        backbone, model = models[model_name]['model']
        backbone.to(device)
        model.to(device)
        optimizer = AdamW(model.parameters(), weight_decay=1e-8, lr=1e-5)
        loss_function = torch.nn.MSELoss()
        if models[model_name]['use_masks']:
            loss_function = iou
        losses = {
            'train': [],
            'val': []
        }
        for epoch in range(num_epochs):
            # Train the model
            model.train()
            # Setup progress bars
            dataloader = dataloaders['train'][0](*dataloaders['train'][1])
            progress_bar_train = tqdm(dataloader)
            progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: TBD")
            # Store the total loss
            total_loss_train = 0
            for batch, targets, masks, _ in progress_bar_train:
                # If masks are the targets, update the ground truth
                if models[model_name]['use_masks']:
                    targets = masks
                # Load batch into memory
                batch = batch.to(device)
                targets = targets.to(device)
                # Make predictions
                # First apply the backbone with no learning
                with torch.no_grad():
                    batch_of_segments = backbone(batch).to(device)
                predictions = model(batch_of_segments)
                # Compute loss
                loss = loss_function(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                # Save loss for printouts
                total_loss_train += torch.mean(loss).item()
                losses['train'].append(loss.item())
                progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {round(loss.item(), 3)} |"
                                                   f" WORST LOSS: {round(np.max(losses['train']), 3)} |"
                                                   f" RUNNING LOSS: {round(np.mean(losses['train'][max(0, len(losses['train'])-20):]), 3)} |")
                # Learn
                optimizer.step()

            total_loss_train /= len(progress_bar_train)

            # Validate the model
            model.train(False)
            dataloader = dataloaders['val'][0](*dataloaders['val'][1])
            progress_bar_val = tqdm(dataloader)
            progress_bar_val.set_description(f"[VALIDATION] | EPOCH {epoch}")
            total_loss_val = 0
            for batch, targets, masks, _ in progress_bar_val:
                if models[model_name]['use_masks']:
                    targets = masks
                batch = batch.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    batch_of_segments = backbone(batch)
                    predictions = model(batch_of_segments)
                    loss = loss_function(predictions, targets)
                    total_loss_val += torch.mean(loss).item()
                    losses['val'].append(loss.item())
            total_loss_val /= len(progress_bar_val)
            print(
                f'EPOCH {epoch} | TOTAL TRAINING LOSS: {round(total_loss_train, 3)} | TOTAL VALIDATION LOSS: {round(total_loss_val, 3)}')
            path = os.path.join(models[model_name]['path'], f"model_{saved_time}_{model_name}_{epoch}.model")
            torch.save(model, path)
            os.makedirs(os.path.join(models[model_name]['path'], 'stats'), exist_ok=True)
            with open(os.path.join(models[model_name]['path'], 'stats', f"model_{saved_time}_{model_name}_{epoch}.json"),
                      'w') as file:
                json.dump(losses, file)
            print(f"MODEL SAVED IN {path}")

        # Test the model
        model.train(False)
        dataloader = dataloaders['test'][0](*dataloaders['test'][1])
        progress_bar_test = tqdm(dataloader)
        progress_bar_test.set_description(f"[TESTING]")
        total_loss_test = 0
        for batch, targets, masks, _ in progress_bar_test:
            if models[model_name]['use_masks']:
                targets = masks
            batch = batch.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                batch_of_segments = backbone(batch)
                predictions = model(batch_of_segments)
                loss = loss_function(predictions, targets)
                total_loss_test += torch.mean(loss).item()
        total_loss_test /= len(progress_bar_test)
        print(f"[RESULTS] TOTAL TEST LOSS: {round(total_loss_test, 3)}")


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
    num_epochs = 15
    batch_size = 60
    culane_dataloader = {
        'train': (get_dataloader, ('train', batch_size, 10, False, False, True)),
        'val': (get_dataloader, ('val', batch_size, 100, False, False, True)),
        'test': (get_dataloader, ('test', batch_size, 100, False, False, True))
    }
    models = {
        "vitt": {"model": get_vitt(device), "path": "./models/checkpoints/vitt/", "use_masks": False}
    }

    training_loop(num_epochs, culane_dataloader, models, device)
