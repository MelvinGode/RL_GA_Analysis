# test the trained q-table with the environment
import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

SEED = 1714389008

def get_discrete_state(state, bins, obsSpaceSize):
	stateIndex = []
	for i in range(obsSpaceSize):
		stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
	return tuple(stateIndex)


os.chdir(os.path.dirname(os.path.abspath(__file__)))



env = gym.make('CartPole-v1', render_mode='rgb_array')
np.random.seed(SEED)
# Load the q-table
def load_q_table(qTablePath):
    qTable = np.load(qTablePath, allow_pickle=True)
    numBins = 20
    obsSpaceSize = len(env.observation_space.high)

    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-.418, .418, numBins),
        np.linspace(-4, 4, numBins)
    ]

    return qTable, bins, obsSpaceSize

if len(sys.argv) > 1:
  qTablePath = sys.argv[1]
  qTable, bins, obsSpaceSize  = load_q_table(qTablePath)
else:
  qTable, bins, obsSpaceSize  = load_q_table('data/qTable_1714389008.npy')



# Get the size of each buc

def test_q_table(qTable, env,  numEpisodes=100,  seed = SEED,bins = bins, obsSpaceSize = obsSpaceSize):
    # Test the trained q-table with the environment
    for episode in range(numEpisodes):
        state = env.reset(seed=seed)
        done = False
        steps = 0
        discrete_state = get_discrete_state(env.observation_space.high, bins, obsSpaceSize)
        while not done and steps < 500:
            action = np.argmax(qTable[discrete_state])
            state, reward, done, _, info = env.step(action)
            discrete_state = get_discrete_state(state, bins, obsSpaceSize)
            steps += reward

        print(f'Episode {episode} reward: {steps}')

#def test_GA()

test_q_table(qTable, env)