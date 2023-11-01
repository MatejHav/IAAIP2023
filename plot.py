import time

import numpy as np
import json
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from main import get_dataloader


def plot_one_epoch(path, epoch, label, median_label, title, x_axis, y_axis):
    stat = json.load(open(path.format(epoch=epoch), 'r'))
    y = stat['train']
    plt.plot(y, label=label.format(mode='Train'))
    plt.plot([np.median(stat['train'][:i + 1]) for i in range(len(stat['train']))], label=median_label.format(mode='Train'))
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.ylim(0.9*np.min(y), 1.1 * np.max(y))
    plt.title(title.format(mode='Train', epoch=epoch))
    plt.show()

    y = stat['val']
    plt.plot(y, label=label.format(mode='Validation'))
    plt.plot([np.median(stat['val'][:i + 1]) for i in range(len(stat['val']))], label=median_label.format(mode='Validation'))
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.ylim(0.9 * np.min(y), 1.1 * np.max(y))
    plt.title(title.format(mode='Validation', epoch=epoch))
    plt.show()


def plot_epochs(path, train_label, val_label, title, max_epoch, x_axis, y_axis):
    train_stat_mean = []
    train_stat_max = []
    train_stat_std = []
    val_stat_mean = []
    val_stat_max = []
    val_stat_std = []
    for epoch in range(max_epoch):
        stats = json.load(open(path.format(epoch=epoch), 'r'))
        train_stat_mean.append(np.mean(stats['train']))
        val_stat_mean.append(np.mean(stats['val']))
        train_stat_max.append(np.max(stats['train']))
        val_stat_max.append(np.max(stats['val']))
        train_stat_std.append(np.mean(stats['train']))
        val_stat_std.append(np.mean(stats['val']))
    train_stat_mean = np.array(train_stat_mean)
    train_stat_max = np.array(train_stat_max)
    train_stat_std = np.array(train_stat_std)
    val_stat_mean = np.array(val_stat_mean)
    val_stat_max = np.array(val_stat_max)
    val_stat_std = np.array(val_stat_std)
    plt.plot(train_stat_mean, label=train_label.format(stat='Mean'))
    # plt.plot(train_stat_max, label=train_label.format(stat='Maximum'))
    # plt.fill_between(range(max_epoch), train_stat_mean - train_stat_std, train_stat_mean + train_stat_std, alpha=0.3)
    plt.plot(val_stat_mean, label=val_label.format(stat='Mean'))
    # plt.plot(val_stat_max, label=val_label.format(stat='Maximum'))
    # plt.fill_between(range(max_epoch), val_stat_std - val_loss_std, val_stat_mean + val_stat_std, alpha=0.3)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    # plt.savefig(f'./results/stat_{int(time.time()*10000)}.jpg')
    plt.show()

def plot_iou_across_frame(model, batch_size, loader, threshold=0.5):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA RECOGNIZED.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS RECOGNIZED.")
    else:
        device = torch.device("cpu")
        print("NO GPU RECOGNIZED.")
    model.to(device)

    scores = np.zeros(batch_size)
    for batch, _, targets, _ in tqdm(loader):
        batch = batch.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            pred = model(batch)
            up = torch.logical_and(pred >= threshold, targets >= threshold).sum(dim=[1, 2])
            down = torch.logical_or(pred >= threshold, targets >= threshold).sum(dim=[1, 2])
            iou = (up + 1e-6) / (down + 1e-6)
            scores += iou.cpu().numpy()
    scores /= len(loader)

    plt.plot(scores)
    plt.title(f'IoU over frames of a video computed with threshold of {threshold}')
    plt.xlabel('Frame number')
    plt.ylabel('Mean IoU')
    plt.show()

if __name__ == '__main__':
    epoch = 0
    max_epochs = 46
    path = "./models/checkpoints/vitt/stats/loss_model_1698863414_vitt_{epoch}.json"
    # plot_one_epoch(path, epoch, '{mode} loss', 'Median {mode} loss', title="{mode} loss of ViTT during epoch {epoch}",
    #                x_axis="batch number", y_axis="Binary Cross Entropy")
    plot_epochs(path, '{stat} Train Loss', '{stat} Validation Loss', "Loss of ViTT over multiple epochs", max_epochs,
                "Epoch number", "Focal Loss poly")
    # IoU
    path = "./models/checkpoints/vitt/stats/iou_model_1698863414_vitt_{epoch}.json"
    # plot_one_epoch(path, epoch, '{mode} IoU', 'Median {mode} IoU', title="IoU on {mode} set of ViTT during epoch {epoch}",
    #                x_axis="batch number",
    #                y_axis="IoU")
    plot_epochs(path, '{stat} Train IoU', '{stat} Validation IoU', "IoU of ViTT over multiple epochs", max_epochs,
                "Epoch number", "IoU")
    # from models.model_collection import get_vitt
    #
    # backbone, model = get_vitt(None)
    # state_dict = torch.load('models/checkpoints/vitt/model_1698730409_vitt_15.model')
    # model.load_state_dict(state_dict)
    # plot_iou_across_frame(model, 32, get_dataloader('train', 32, 10, True))
