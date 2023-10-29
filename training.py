import os
import time
import json

from torch.optim import AdamW
from tqdm.auto import tqdm
from main import *
from models.model_collection import *
from utils import FocalLoss_poly

import torch.nn.functional as F


def iou(predictions, targets):
    """
    Computes 1 - IoU to act as a loss function.

    :param predictions: segmentation mask, must be of the same shape as target containing values [0, 1]
    :param targets: binary segmentation mask of the ground truth
    :return: average IoU ove the elements in the batch
    """
    eps = 1e-5
    return (1 - ((predictions * targets).sum(dim=[1, 2]) + eps) / torch.clamp((predictions + targets + eps), min=0, max=1).sum(
        dim=[1, 2])).mean()


criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.02, 1.02]), reduction='sum')


def ce_loss(pred, tar):
    pred = torch.stack((1 - pred, pred)).view(pred.shape[0], 2, *pred.shape[1:]).to(device)
    tar = torch.stack((1 - tar, tar)).view(tar.shape[0], 2, *tar.shape[1:]).to(device)
    return criterion(pred, tar)


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
        optimizer = AdamW(model.parameters(), weight_decay=1e-10, lr=0.001)
        loss_function = FocalLoss_poly(alpha=0.2, gamma=2, epsilon=0.1, size_average=False).to(device)
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
            for batch, _, targets, id in progress_bar_train:
                optimizer.zero_grad()
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
                if torch.any(torch.isnan(loss)):
                    print(id)
                    exit()
                loss.backward()
                # for param in model.parameters():
                #     print(param.grad)
                # Learn
                optimizer.step()
                # Save loss for printouts
                intersect_over_union = 1 - iou(predictions, targets).item()
                losses['train'].append(loss.item())
                ious['train'].append(intersect_over_union)
                progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {loss.item():.3f} |"
                                                   f" MEDIAN LOSS: {np.median(losses['train']):.3f} |"
                                                   f" RUNNING LOSS: {np.mean(losses['train'][max(0, len(losses['train']) - 100):]):.3f} |"
                                                   f" IOU: {intersect_over_union:.3f} |"
                                                   f" BEST IOU: {max(ious['train']):.3f} |"
                                                   f" RUNNING IOU: {np.mean(ious['train'][max(0, len(ious['train']) - 100):]):.3f} | "
                                                   f" MAX STD ACROSS BATCH: {predictions.std(dim=0).max().item():.3f} | "
                                                   f" MIN AND MAX: {predictions.min().item():.3f}, {predictions.max().item():.3f} | ")

            # Validate the model
            dataloader = dataloaders['val'][0](*dataloaders['val'][1])
            progress_bar_val = tqdm(dataloader)
            progress_bar_val.set_description(f"[VALIDATION] | EPOCH {epoch} | ")
            total_loss_val = 0
            for batch, _, targets, _ in progress_bar_val:
                batch = batch.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    batch_of_segments = backbone(batch)
                    predictions = model(batch_of_segments)
                    loss = loss_function(predictions, targets)
                    total_loss_val += torch.mean(loss).item()
                    losses['val'].append(loss.item())
                    intersect_over_union = 1 - iou(predictions, targets).item()
                    ious['val'].append(intersect_over_union)
            total_loss_val /= len(progress_bar_val)
            print(f'EPOCH {epoch} | TOTAL VALIDATION LOSS: {round(total_loss_val, 3)}')
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

        # Test the model
        model.train(False)
        dataloader = dataloaders['test'][0](*dataloaders['test'][1])
        progress_bar_test = tqdm(dataloader)
        progress_bar_test.set_description(f"[TESTING]")
        total_loss_test = 0
        for batch, _, targets, _ in progress_bar_test:
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
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([0.02, 1.02]), reduction='sum').to(device)

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
