import numpy as np
import json
import torch
import matplotlib.pyplot as plt


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
    train_stat_std = []
    val_stat_mean = []
    val_stat_std = []
    for epoch in range(max_epoch):
        stats = json.load(open(path.format(epoch=epoch), 'r'))
        train_stat_mean.append(np.mean(stats['train']))
        val_stat_mean.append(np.mean(stats['val']))
        train_stat_std.append(np.mean(stats['train']))
        val_stat_std.append(np.mean(stats['val']))
    train_stat_mean = np.array(train_stat_mean)
    train_stat_std = np.array(train_stat_std)
    val_stat_mean = np.array(val_stat_mean)
    val_stat_std = np.array(val_stat_std)
    plt.plot(train_stat_mean, label=train_label)
    # plt.fill_between(range(max_epoch), train_stat_mean - train_stat_std, train_stat_mean + train_stat_std, alpha=0.3)
    plt.plot(val_stat_mean, label=val_label)
    # plt.fill_between(range(max_epoch), val_stat_std - val_loss_std, val_stat_mean + val_stat_std, alpha=0.3)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    epoch = 6
    path = "./models/checkpoints/vitt/stats/loss_model_1698619875_vitt_{epoch}.json"
    plot_one_epoch(path, epoch, '{mode} loss', 'Median {mode} loss', title="{mode} loss of ViTT during epoch {epoch}",
                   x_axis="batch number", y_axis="Binary Cross Entropy")
    plot_epochs(path, 'Mean Train Loss', 'Mean Validation Loss', "Loss of ViTT over multiple epochs", 7,
                "Epoch number", "Binary Cross Entropy")
    # IoU
    path = "./models/checkpoints/vitt/stats/iou_model_1698619875_vitt_{epoch}.json"
    plot_one_epoch(path, epoch, '{mode} IoU', 'Median {mode} IoU', title="IoU on {mode} set of ViTT during epoch {epoch}",
                   x_axis="batch number",
                   y_axis="IoU")
    plot_epochs(path, 'Mean Train IoU', 'Mean Validation IoU', "IoU of ViTT over multiple epochs", 7,
                "Epoch number", "IoU")
