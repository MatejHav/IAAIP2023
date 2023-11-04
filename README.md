# IAAIP2023
Repository for the Interdisciplinary Advanced AI Project for the school year 2023/2024

# Instalation

1. Install the required packages using ```pip install -r requirements.txt```

## Data
1. Download the [CULane database](https://xingangpan.github.io/projects/CULane.html) into `./culane/data/`
2. Run the `./culane/transform_ground_truth.py` file. This will create formatted ground truths for the files. This will take quite a while (around 1h on my desktop).
3. Run the `./culane/main.py`. Initially this will create a cache file in `./culane/cache/` which might take a while. When running it again, the cache will be loaded.
4. Use button 0 to go through the images and check the ground truth.

## Running the model
1. In ```training.py``` you can find the settings that you can change. Namely the batch_size and how much of the data do you want to use.
2. After setting up the necessary settings, you can run ```python training.py``` to run the model, which will inform you throughout about its progress.
3. Afterwards, you can find the model checkpoints in the directory you've setup. Copy the latest model and put it into ```main.py``` to visualize the results.
4. If you want to plot the loss and IoU across multiple epochs, copy the number assigned to the model checkpoints (the number in model_{number}_{epoch}) and replace it in ```plot.py```. You can then run this file and plot the results of your run.


