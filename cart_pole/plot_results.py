import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def plot_results(ga_time,ga_results, rl_time, rl_results, x_label, y_label, title):
    # Plot average/seconds
    plt.plot(ga_time, ga_results, label='GA')
    plt.plot(rl_time, rl_results, label='RL')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(f'plots/RL_GA_comparison_{time.time()}.png')
    plt.show()

# gets paths as arguments
if len(sys.argv) > 1:
    GA_filename = sys.argv[1]
    RL_filename = sys.argv[2]
else:
    GA_filename = 'GA_v2_metrics_+100_fitness_0.005_fitness_2_10.npy'
    RL_filename = 'metrics_1715514318.npy'

# Load data from GA/data
ga_data = np.load(f'GA/data/{GA_filename}', allow_pickle=True).item()
rl_data = np.load(f'RL/data/{RL_filename}', allow_pickle=True).item()

plot_results(ga_data['time'],ga_data['max'] , rl_data['time'],rl_data['max'], 'Time [s]', 'Max Reward', 'Maximum Reward over Time')
plot_results(ga_data['time'],ga_data['avg'] , rl_data['time'],rl_data['avg'], 'Time [s]', 'Average Reward', 'Average Reward over Time')