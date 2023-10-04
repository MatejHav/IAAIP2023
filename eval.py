import gc
import os
import torch
from main import get_dataloader
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from IPython.display import display

from models.backbone.backbone import Backbone


def compute_iou(prediction, target):
    prediction = prediction.cpu()
    target = target.cpu()
    img_height = 320
    img_width = 800
    # Covert predictions and targets into mask
    pred_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    tar_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for y in range(prediction.shape[0]):
        for x in range(prediction.shape[1]):
            if prediction[y, x, 2] >= 0.5:
                length = prediction[y, x, 0]
                angle = prediction[y, x, 1]
                temp = [(int(x / prediction.shape[1] * img_width), int(y / prediction.shape[0] * img_height)),
                        (int((x / prediction.shape[1] + np.cos(angle) * length) * img_width),
                         int((y / prediction.shape[0] + np.sin(angle) * length) * img_height))]
                pred_img = cv2.line(pred_img, temp[0], temp[1], color=(1, 1, 1), thickness=3)
            if target[y, x, 2] >= 0.5:
                length = target[y, x, 0]
                angle = target[y, x, 1]
                temp = [(int(x / prediction.shape[1] * img_width), int(y / prediction.shape[0] * img_height)),
                        (int((x / prediction.shape[1] + np.cos(angle) * length) * img_width),
                         int((y / prediction.shape[0] + np.sin(angle) * length) * img_height))]
                tar_img = cv2.line(tar_img, temp[0], temp[1], color=(1, 1, 1), thickness=3)
    # Compute IOU over mask
    intersection = pred_img * tar_img
    union = np.clip(pred_img + tar_img, a_min=0, a_max=1)
    return sum(intersection) / sum(union)


def show_culane_statistics(models, show=False, save=True, root='./', batch_size=32, device=torch.device,
                           filename='results.csv'):
    dataloaders = {
        'normal': get_dataloader('normal', batch_size),
        'crowd': get_dataloader('crowd', batch_size),
        'hlight': get_dataloader('hlight', batch_size),
        'shadow': get_dataloader('shadow', batch_size),
        'noline': get_dataloader('noline', batch_size),
        'arrow': get_dataloader('arrow', batch_size),
        'curve': get_dataloader('curve', batch_size),
        'cross': get_dataloader('cross', batch_size),
        'night': get_dataloader('night', batch_size)
    }
    result_table = pd.DataFrame(columns=list(dataloaders.keys()), index=list(map(lambda x: x['name'], models)))
    for model_dict in models:
        name = model_dict['name']
        backbone = Backbone(model_dict['backbone'])
        backbone.to(device)
        model = torch.load(model_dict['path'])
        model.to(device)
        row = {}
        total_batch_len = sum([len(loader) for loader in dataloaders.values()])
        bar = tqdm(range(len(dataloaders.keys()) * total_batch_len))
        for key, dataloader in dataloaders.items():
            bar.set_description(f"[EVALUATION] CURRENT CATEGORY: {key} | ")
            all_iou = []
            for batch, targets, _ in dataloader:
                # Load data into GPU
                batch = batch.to(device)
                targets = targets.to(device)
                # make predictions
                with torch.no_grad():
                    batch_of_segments = backbone(batch)
                    predictions = model(batch_of_segments)
                # Clean memory
                del batch, batch_of_segments
                gc.collect()
                torch.cuda.empty_cache()
                # Compute IOU
                for frame_num in range(predictions.shape[0]):
                    all_iou.append(compute_iou(predictions[frame_num], targets[frame_num]))
                bar.update(1)
                # Clean memory
                del predictions, targets
                gc.collect()
                torch.cuda.empty_cache()
            row[key] = np.mean(all_iou)

        result_table.append(pd.Series(row, index=name, name=name))
    if save:
        path = os.path.join(root, filename)
        result_table.to_csv(path)
        print(f'RESULTS STORED IN {path}')
    if show:
        display(result_table)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA RECOGNIZED.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS RECOGNIZED.")
    else:
        device = torch.device("cpu")
        print("NO GPU RECOGNIZED.")
    models = [
        {'path': './models/checkpoints/model_1696365334_9.model', 'name': 'basic_lane_transformer',
         'backbone': 'resnet18'}
    ]
    root = './results/'
    os.makedirs(root, exist_ok=True)
    show_culane_statistics(models, show=True, save=True, root=root, batch_size=32, device=device)
