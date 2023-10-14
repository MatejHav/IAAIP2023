# IAAIP2023
Repository for the Interdisciplinary Advanced AI Project for the school year 2023/2024

# Instalation

## Data

### CULane
1. Download the [CULane database](https://xingangpan.github.io/projects/CULane.html) into `./culane/data/`
2. Run the `./culane/transform_ground_truth.py` file. This will create formatted ground truths for the files. This will take quite a while (around 1h on my desktop).
3. Run the `./culane/main.py`. Initially this will create a cache file in `./culane/cache/` which might take a while. When running it again, the cache will be loaded.
4. Use button 0 to go through the images and check the ground truth.