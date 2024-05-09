import numpy as np
import os

# change working directory to the current folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# generate 10k random seeds
SEEDS = np.random.randint(0, 2**32, 20000)

#save them to a file
np.save('data/random_seeds.npy', SEEDS)