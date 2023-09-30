import torch
import os
import time

from torch.optim import AdamW
from tqdm.auto import tqdm
from culane.main import get_dataloader
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
    selection_lane = targets[:, :, :, 2] > 0.5
    should_be_empty = (predictions[selection_empty] - targets[selection_empty]) ** 2 / torch.sum(selection_empty)
    should_be_lane = (predictions[selection_lane] - targets[selection_lane]) ** 2 / torch.sum(selection_lane)
    return 0.5 * should_be_lane.sum() + 0.5 * should_be_empty.sum()


def training_loop(num_epochs, dataloaders, models, device):
    for model_name in models:
        print("\n" + ''.join(['#'] * 25) + "\n")
        print(f'TRAINING MODEL {model_name}. SAVING INTO {models[model_name]["path"]}.')
        if not os.path.exists(models[model_name]['path']):
            os.mkdir(models[model_name]['path'])
        model = models[model_name]['model']
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=5e-5)
        for epoch in range(num_epochs):
            # Train the model
            model.train()
            progress_bar_train = tqdm(dataloaders['train'])
            progress_bar_train.set_description(f"[TRAINING] | EPOCH {epoch}")
            total_loss_train = 0
            for batch, targets, _ in progress_bar_train:
                batch = batch.to(device)
                targets = targets.to(device)
                predictions = model(batch)
                loss = compute_loss(predictions, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss_train += torch.mean(loss)
            total_loss_train /= len(progress_bar_train)

            # Validate the model
            model.train(False)
            progress_bar_val = tqdm(dataloaders['val'])
            progress_bar_val.set_description(f"[VALIDATION] | EPOCH {epoch}")
            total_loss_val = 0
            for batch, targets, _ in progress_bar_val:
                batch = batch.to(device)
                targets = targets.to(device)
                predictions = model(batch)
                loss = compute_loss(predictions, targets)
                total_loss_val += torch.mean(loss)
            total_loss_val /= len(progress_bar_val)
            print(f'EPOCH {epoch} | TOTAL TRAINING LOSS: {total_loss_train} | TOTAL VALIDATION LOSS: {total_loss_val}')

        # Test the model
        model.train(False)
        progress_bar_test = tqdm(dataloaders['test'])
        progress_bar_test.set_description(f"[TESTING]")
        total_loss_test = 0
        for batch, targets, _ in progress_bar_test:
            batch = batch.to(device)
            targets = targets.to(device)
            predictions = model(batch)
            loss = compute_loss(predictions, targets)
            total_loss_test += torch.mean(loss)
        total_loss_test /= len(progress_bar_test)
        print(f"TOTAL TEST LOSS: {total_loss_test}")
        path = os.path.join(models[model_name]['path'], f"model_{time.time()}.model")
        torch.save(model, path)
        print(f"MODEL SAVED IN {path}.")


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA RECOGNIZED.")
    else:
        print("CUDA NOT RECOGNIZED. USING CPU INSTEAD.")

    # Training Parameters
    num_epochs = 20
    batch_size = 30
    culane_dataloader = {
        'train': get_dataloader('train', batch_size),
        'val': get_dataloader('val', batch_size),
        'test': get_dataloader('test', batch_size)
    }
    models = {
        "basic_lane_detector": {"model": get_basic_model(device), "path": "./models/checkpoints/"}
    }

    training_loop(num_epochs, culane_dataloader, models, device)
