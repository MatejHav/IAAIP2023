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
    plt.plot([np.median(stat['train'][:i + 1]) for i in range(len(stat['train']))],
             label=median_label.format(mode='Train'))
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.ylim(0.9 * np.min(y), 1.1 * np.max(y))
    plt.title(title.format(mode='Train', epoch=epoch))
    plt.show()

    y = stat['val']
    plt.plot(y, label=label.format(mode='Validation'))
    plt.plot([np.median(stat['val'][:i + 1]) for i in range(len(stat['val']))],
             label=median_label.format(mode='Validation'))
    plt.legend()
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.ylim(0.9 * np.min(y), 1.1 * np.max(y))
    plt.title(title.format(mode='Validation', epoch=epoch))
    plt.show()

def compute_iou(pred, tar, threshold=0.5):
    up = torch.logical_and(pred >= threshold, tar >= 0.5).sum(dim=[1, 2])
    down = torch.logical_or(pred >= threshold, tar >= 0.5).sum(dim=[1, 2])
    return ((up + 1e-6) / (down + 1e-6)).mean().item()

def compute_accuracy(pred, tar, threshold=0.5):
    true_positives = torch.logical_and(pred >= threshold, tar >= 0.5).sum(dim=[1, 2])
    true_negatives = torch.logical_and(pred < threshold, tar < 0.5).sum(dim=[1, 2])
    return ((true_positives + true_negatives + 1e-6) / (np.prod(pred.shape[1:]) + 1e-6)).mean().item()

def compute_precision(pred, tar, threshold=0.5):
    true_positives = torch.logical_and(pred >= threshold, tar >= 0.5).sum(dim=[1, 2])
    false_positives = torch.logical_and(pred >= threshold, tar < 0.5).sum(dim=[1, 2])
    return ((true_positives + 1e-6) / (true_positives + false_positives + 1e-6)).mean().item()

def compute_recall(pred, tar, threshold=0.5):
    true_positives = torch.logical_and(pred >= threshold, tar >= 0.5).sum(dim=[1, 2])
    false_negatives = torch.logical_and(pred < threshold, tar >= 0.5).sum(dim=[1, 2])
    return ((true_positives + 1e-6) / (true_positives + false_negatives + 1e-6)).mean().item()

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


def plot_iou_across_frame(model, batch_size, loader, training=False, threshold=0.5):
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
    model.training = training

    scores = np.zeros(batch_size)
    for batch, targets in tqdm(loader):
        batch = batch.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            pred = model(batch, targets)
            up = torch.logical_and(pred >= threshold, targets >= threshold).sum(dim=[1, 2])
            down = torch.logical_or(pred >= threshold, targets >= threshold).sum(dim=[1, 2])
            iou = (up + 1e-6) / (down + 1e-6)
            scores += iou.cpu().numpy()
    scores /= len(loader)

    plt.plot(scores)
    plt.title(f'IoU over frames with threshold of {threshold}, training set to {training}')
    plt.xlabel('Frame number')
    plt.ylabel('Mean IoU')
    plt.show()

def iou_through_thresholds(model, batch_size, number_of_thresholds):
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
    loader = get_dataloader('test', batch_size, 100, True)
    pbar = tqdm(loader)
    model.training = False
    results = np.zeros(number_of_thresholds)

    for i, (images, masks) in enumerate(pbar):
        # RUNNING TRAINED MODEL PREDICTIONS
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            labels = model(images, masks)
        for j in range(number_of_thresholds):
            results[j] += compute_iou(labels, masks, threshold=j / number_of_thresholds) / len(pbar)
    plt.plot([i / number_of_thresholds for i in range(number_of_thresholds)], results)
    plt.xlabel('Threshold for IoU')
    plt.ylabel('Mean IoU')
    plt.title('Change of IoU score over different thresholds on the test set')
    plt.show()

def metrics_on_test_set(model, batch_size):
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
    loader = get_dataloader('test', batch_size, 100, True)
    pbar = tqdm(loader)
    model.training = False
    iou = []
    accuracy = []
    f1 = []
    precision = []
    recall = []

    for i, (images, masks) in enumerate(pbar):
        # RUNNING TRAINED MODEL PREDICTIONS
        images = images.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            labels = model(images, masks)
        iou.append(compute_iou(labels, masks, threshold=0.5))
        accuracy.append(compute_accuracy(labels, masks, threshold=0.5) / len(pbar))
        pred_precision = compute_precision(labels, masks, threshold=0.5)
        pred_recall = compute_recall(labels, masks, threshold=0.5)
        pred_f1 = (pred_precision * pred_recall + 1e-6) / (pred_precision + pred_recall + 1e-6)
        precision.append(pred_precision / len(pbar))
        recall.append(pred_recall / len(pbar))
        f1.append(pred_f1 / len(pbar))

    print(f"FINAL RESULTS:\nIoU: {np.mean(iou)}\nACCURACY: {np.mean(accuracy)}\nPRECISION: {np.mean(precision)}\nRECALL: {np.mean(recall)}\nF1: {np.mean(f1)}")


if __name__ == '__main__':
    # max_epochs = 7
    # path = "./models/checkpoints/vitt/stats/loss_model_1699126061_vitt_{epoch}.json"
    # plot_epochs(path, '{stat} Train Loss', '{stat} Validation Loss', "Loss of ViTT over multiple epochs", max_epochs,
    #             "Epoch number", "1 - soft IoU")
    #
    # path = "./models/checkpoints/vitt/stats/iou_model_1699126061_vitt_{epoch}.json"
    # plot_epochs(path, '{stat} Train IoU', '{stat} Validation IoU', "IoU of ViTT over multiple epochs", max_epochs,
    #             "Epoch number", "IoU")
    from models.model_collection import get_vitt

    model = get_vitt(None)
    state_dict = torch.load('models/checkpoints/vitt/model_1699126061_vitt_6.model')
    model.load_state_dict(state_dict)
    # plot_iou_across_frame(model, 15, get_dataloader('test', 15, 100, True), training=False)
    metrics_on_test_set(model, 15)
    # iou_through_thresholds(model, 15, 35)
