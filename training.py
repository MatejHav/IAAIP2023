import torch
import os
import time
import gc
import numpy as np

from torch.optim import AdamW
from tqdm.auto import tqdm
from main import get_dataloader
from models.model_collection import get_basic_model


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    In this loss function we compute the MSE loss across all the predictions.
    However, to avoid the network to just output no lanes at all,
    as that is the easier option with way more many empty cells than lane cells,
    we weight the different outputs such that the network has the same weight on lane cells and empty cells.
    :param predictions: Predictions of the network.
    :param targets: Ground truth.
    :return: Total loss
    """
    selection_empty = targets[:, :, :, 2] <= 0.5
    should_be_empty = (predictions[selection_empty] - targets[selection_empty]) ** 2 / torch.sum(selection_empty)
    temp = 0.5 * should_be_empty.sum()
    selection_lane = targets[:, :, :, 2] > 0.5
    should_be_lane = (predictions[selection_lane] - targets[selection_lane]) ** 2 / torch.sum(selection_lane)
    return temp + 0.5 * should_be_lane.sum()


def training_loop(num_epochs, dataloaders, models, device):
    for model_name in models:
        print("\n" + ''.join(['#'] * 25) + "\n")
        print(f'TRAINING MODEL {model_name}. SAVING INTO {models[model_name]["path"]}.')
        if not os.path.exists(models[model_name]['path']):
            os.mkdir(models[model_name]['path'])
        backbone, model = models[model_name]['model']
        backbone.to(device)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        loss_function = compute_loss
        for epoch in range(num_epochs):
            # Train the model
            model.train()
            progress_bar_train = tqdm(dataloaders['train'])
            progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: TBD")
            total_loss_train = 0
            for batch, targets, _ in progress_bar_train:
                # Load batch into memory
                batch = batch.to(device)
                targets = targets.to(device)
                # Make predictions
                # Turn all the frames into segments. Shape(batch_size, segment_number, segment_x, segment_y)
                with torch.no_grad():
                    batch_of_segments = backbone(batch).to(device)
                predictions = model(batch_of_segments)
                del batch, batch_of_segments
                gc.collect()
                torch.cuda.empty_cache()
                # Compute loss
                loss = loss_function(predictions, targets)
                del targets, predictions
                gc.collect()
                torch.cuda.empty_cache()
                loss.backward()
                # Save loss for printouts
                total_loss_train += torch.mean(loss).item()
                progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch} | LOSS: {round(loss.item(), 3)}")
                del loss
                gc.collect()
                torch.cuda.empty_cache()
                # Learn
                optimizer.step()
                optimizer.zero_grad()

            total_loss_train /= len(progress_bar_train)

            # Validate the model
            model.train(False)
            progress_bar_val = tqdm(dataloaders['val'])
            progress_bar_val.set_description(f"[VALIDATION] | EPOCH {epoch}")
            total_loss_val = 0
            for batch, targets, _ in progress_bar_val:
                batch = batch.to(device)
                targets = targets.to(device)
                with torch.no_grad():
                    batch_of_segments = backbone(batch).to(device)
                    predictions = model(batch_of_segments)
                    del batch, batch_of_segments
                    gc.collect()
                    torch.cuda.empty_cache()
                    loss = compute_loss(predictions, targets)
                    del targets, predictions
                    gc.collect()
                    torch.cuda.empty_cache()
                    total_loss_val += torch.mean(loss).item()
                    del loss
                    gc.collect()
                    torch.cuda.empty_cache()
            total_loss_val /= len(progress_bar_val)
            print(f'EPOCH {epoch} | TOTAL TRAINING LOSS: {round(total_loss_train, 3)} | TOTAL VALIDATION LOSS: {round(total_loss_val, 3)}')

        # Test the model
        model.train(False)
        progress_bar_test = tqdm(dataloaders['test'])
        progress_bar_test.set_description(f"[TESTING]")
        total_loss_test = 0
        for batch, targets, _ in progress_bar_test:
            batch = batch.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                batch_of_segments = backbone(batch).to(device)
                predictions = model(batch_of_segments)
                del batch, batch_of_segments
                gc.collect()
                torch.cuda.empty_cache()
                loss = compute_loss(predictions, targets)
                del targets, predictions
                gc.collect()
                torch.cuda.empty_cache()
                total_loss_test += torch.mean(loss).item()
                del loss
                gc.collect()
                torch.cuda.empty_cache()
        total_loss_test /= len(progress_bar_test)
        print(f"TOTAL TEST LOSS: {round(total_loss_test, 3)}")
        path = os.path.join(models[model_name]['path'], f"model_{time.time()}.model")
        torch.save(model, path)
        print(f"MODEL SAVED IN {path}.")


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
    num_epochs = 5
    batch_size = 32
    culane_dataloader = {
        'train': get_dataloader('train', batch_size, subset=10),
        'val': get_dataloader('val', batch_size),
        'test': get_dataloader('test', batch_size)
    }
    models = {
        "basic_lane_detector": {"model": get_basic_model(device), "path": "./models/checkpoints/"}
    }

    training_loop(num_epochs, culane_dataloader, models, device)
