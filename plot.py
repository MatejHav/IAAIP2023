import numpy as np
import json
import torch
import matplotlib.pyplot as plt

def plot_train_loss_epoch(path, title, x_axis, y_axis):
    losses = json.load(open(path, 'r'))
    plt.plot(losses['train'], label='Train loss')
    plt.plot([np.median(losses['train'][:i+1]) for i in range(len(losses['train']))], label='Median Train Loss')
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()

def plot_epoch_loss(path, title, max_epoch, x_axis, y_axis):
    train_loss_mean = []
    train_loss_std = []
    val_loss_mean = []
    val_loss_std = []
    for epoch in range(max_epoch):
        losses = json.load(open(path.format(epoch=epoch), 'r'))
        train_loss_mean.append(np.mean(losses['train']))
        val_loss_mean.append(np.mean(losses['val']))
        train_loss_std.append(np.mean(losses['train']))
        val_loss_std.append(np.mean(losses['val']))
    train_loss_mean = np.array(train_loss_mean)
    train_loss_std = np.array(train_loss_std)
    val_loss_mean = np.array(val_loss_mean)
    val_loss_std = np.array(val_loss_std)
    plt.plot(train_loss_mean, label='Mean Train Loss')
    # plt.fill_between(range(max_epoch), train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.3)
    plt.plot(val_loss_mean, label='Mean Validation Loss')
    # plt.fill_between(range(max_epoch), val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.3)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    path = "./models/checkpoints/vitt/stats/model_1697549353_vitt_{epoch}.json"
    # plot_train_loss_epoch(path, title="Training loss of ViTT during epoch 0", x_axis="batch number", y_axis="Binary Cross Entropy")
    plot_epoch_loss(path, "Loss of ViTT over multiple epochs", 9, "Epoch number", "Binary Cross Entropy")