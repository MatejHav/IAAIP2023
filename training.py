import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformer.basic_transformer import BasicTransformer
import dataloader as dl

num_training_steps = 3
num_epochs = 20
train_dataloader = ["this data", "needs to be loaded properly"]
lr_scheduler = "this needs a scheduler"
batch_size = 1

model = BasicTransformer()

dataloader = torch.utils.data.DataLoader(
        dl.CULaneDataset(dataset_path='/home/charles/Desktop/IAAIP-Transformer/culane/data/list', data_list='train_gt', transform=None),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
optimizer = AdamW(model.parameters(), lr=5e-5)
model.to(device)
print(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)