import gc
import os
import torch
import torchvision

from main import get_dataloader
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from IPython.display import display

from models.backbone.backbone import Backbone
from models.model_collection import get_vitt


def transform_output_to_mask(y, x, output, img):
    length = output[y, x, 0]
    angle = output[y, x, 1]

    x1 = (x / output.shape[1] * img.shape[1]).astype(int)
    y1 = (y / output.shape[0] * img.shape[0]).astype(int)

    x2 = ((x / output.shape[1] + np.cos(angle) * length) * img.shape[1]).astype(int)
    y2 = ((y / output.shape[0] + np.sin(angle) * length) * img.shape[0]).astype(int)

    temp = np.column_stack((x1, y1, x2, y2))

    for x1, y1, x2, y2 in temp:
        cv2.line(img, (x1, y1), (x2, y2), color=(1, 1, 1), thickness=3)


def compute_iou(prediction, target):
    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()
    img_height = 320
    img_width = 800
    # Covert predictions and targets into mask
    pred_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    tar_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    # Add predicted lanes into mask
    pred_y, pred_x = np.where(prediction[:, :, 2] >= 0.5)
    transform_output_to_mask(pred_y, pred_x, prediction, pred_img)
    tar_y, tar_x = np.where(target[:, :, 2] >= 0.5)
    transform_output_to_mask(tar_y, tar_x, target, tar_img)
    # Cast into a binary image
    pred_img = pred_img.astype(bool)
    tar_img = tar_img.astype(bool)
    # Compute IOU over mask
    intersection = pred_img & tar_img
    union = pred_img | tar_img
    if np.sum(union) == 0:
        return 1
    return np.sum(intersection) / np.sum(union)

def iou(predictions, targets):
    return (predictions * targets).sum() / torch.clamp((predictions + targets + 1e-5), min=0, max=1).sum()

def show_culane_statistics(models, show=False, save=True, root='./', batch_size=30, device=torch.device,
                           filename='results.csv'):
    dataloaders = {
        'normal': get_dataloader('normal', batch_size, save_fit=True),
        'crowd': get_dataloader('crowd', batch_size, save_fit=True),
        'hlight': get_dataloader('hlight', batch_size, save_fit=True),
        'shadow': get_dataloader('shadow', batch_size, save_fit=True),
        'noline': get_dataloader('noline', batch_size, save_fit=True),
        'arrow': get_dataloader('arrow', batch_size, save_fit=True),
        'curve': get_dataloader('curve', batch_size, save_fit=True),
        'cross': get_dataloader('cross', batch_size, save_fit=True),
        'night': get_dataloader('night', batch_size, save_fit=True)
    }
    result_table = pd.DataFrame(columns=list(dataloaders.keys()), index=list(map(lambda x: x['name'], models)))
    for model_dict in models:
        name = model_dict['name']
        backbone, model = get_vitt(device)
        model.load_state_dict(torch.load(model_dict['path']))
        model.to(device)
        row = {}
        total_batch_len = sum([len(loader) for loader in dataloaders.values()])
        bar = tqdm(range(total_batch_len))
        for key, dataloader in dataloaders.items():
            bar.set_description(f"[EVALUATION] CURRENT CATEGORY: {key} | ")
            all_iou = []
            for batch, targets, mask, _ in dataloader:
                # Load data into GPU
                batch = batch.to(device)
                targets = mask.to(device)
                # make predictions
                with torch.no_grad():
                    batch_of_segments = backbone(batch)
                    predictions = torch.nn.Sigmoid()(model(batch_of_segments))
                # Compute IOU
                for frame_num in range(predictions.shape[0]):
                    all_iou.append(iou(predictions[frame_num], targets[frame_num]).cpu())
                bar.update(1)
            row[key] = np.mean(all_iou)
        result_table.loc[name] = pd.Series(row)
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
    transforms = torchvision.transforms.Resize(size=(576, 576))
    models = [
        {'path': './models/checkpoints/vitt/model_1697726243_vitt_91.model', 'name': 'vitt',
         'backbone': torch.nn.Sequential(transforms, torch.load('./models/backbone/encoder.model'))}
    ]
    root = './results/'
    os.makedirs(root, exist_ok=True)
    show_culane_statistics(models, show=True, save=True, root=root, batch_size=30, device=device)
