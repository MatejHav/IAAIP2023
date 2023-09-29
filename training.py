import torch
from sympy.physics.units import time
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from models.transformer.basic_transformer import BasicTransformer
import dataloader as dl
from culane import backbone
from models.positional_encoder.positional_encoder import PositionalEncoder
from models.basic_lane_detector import BasicLaneDetector

#All these params below definitely need to be replaced with actual values
num_training_steps = 3
num_epochs = 20
#i assume the dataloader will provide everything in batches, determined by batch_size below
train_dataloader = ["this data", "needs to be loaded properly"]
batch_size = 1
# lr_scheduler = "this needs a scheduler probably for later on? doesn't seem too necessary"

#setup
backbone = backbone.ResNet18Backbone()
pe = PositionalEncoder((7, 7), 0.2,
                       512)  # number of segments when using ResNet18 is 512 per image and dimensions are actually 7x7 ?
transformer = BasicTransformer(d_model=49, nhead=7)
model = BasicLaneDetector(backbone, pe, transformer)

print(model.forward(torch.randn(32, 3, 224, 224)).shape)

# model = BasicTransformer()

#This needs a change probably, don't think it's correct on the definition
dataloader = torch.utils.data.DataLoader(
        dl.CULaneDataset(dataset_path='/home/charles/Desktop/IAAIP-Transformer/culane/data/list', data_list='train_gt', transform=None),
        batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True)

#Setting up smaller stuff
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
optimizer = AdamW(model.parameters(), lr=5e-5)
model.to(device)
print(device)
progress_bar = tqdm(range(num_training_steps))
#loss
loss_criterion = nn.NLLLoss()
total_loss = 0
#activating training
model.train()
#keeping track of time
start_time = time.time()
for epoch in range(num_epochs):
    #here im assuming somehow the batches of training data and their anchor targets are given separately for the loss
    for batch, targets in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # THE BELOW IS AN EXAMPLE OF WHAT YOU USE ** for in function arguments, I didn't know this so i included
        # https://blog.enterprisedna.co/what-does-double-star-mean-in-python/ (go to section 2)
        outputs = model(**batch)

        loss = loss_criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        total_loss += loss


        #Below is some code i found for logging, planning to make this work once there is actual data in the pipeline

        # if batch % args.log_interval == 0 and batch > 0:
        #     cur_loss = total_loss / args.log_interval
        #     elapsed = time.time() - start_time
        #     print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
        #           'loss {:5.2f} | ppl {:8.2f}'.format(
        #         epoch, batch, len(train_data) // args.bptt, lr,
        #                       elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()


        #Some code for stopping training early and still getting useful information
        
        # try:
        #     for epoch in range(1, args.epochs + 1):
        #         epoch_start_time = time.time()
        #         train()
        #         val_loss = evaluate(val_data)
        #         print('-' * 89)
        #         print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #               'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                          val_loss, math.exp(val_loss)))
        #         print('-' * 89)
        #         # Save the model if the validation loss is the best we've seen so far.
        #         if not best_val_loss or val_loss < best_val_loss:
        #             with open(args.save, 'wb') as f:
        #                 torch.save(model, f)
        #             best_val_loss = val_loss
        #         else:
        #             # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #             lr /= 4.0
        # except KeyboardInterrupt:
        #     print('-' * 89)
        #     print('Exiting from training early')