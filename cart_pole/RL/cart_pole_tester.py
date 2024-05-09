# test the trained q-table with the environment
import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

SEED = 1445677
NUM_BINS = 20
def get_discrete_state(state, bins, obsSpaceSize):
	stateIndex = []
	for i in range(obsSpaceSize):
		stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
	return tuple(stateIndex)



os.chdir(os.path.dirname(os.path.abspath(__file__)))



env = gym.make('CartPole-v1', render_mode='rgb_array')
np.random.seed(SEED)
# Load the q-table
def load_q_table(qTablePath, numBins=NUM_BINS):
    qTable = np.load(qTablePath, allow_pickle=True)
    
    obsSpaceSize = len(env.observation_space.high)

    bins = [
        np.linspace(-4.8, 4.8, numBins),
        np.linspace(-4, 4, numBins),
        np.linspace(-.418, .418, numBins),
        np.linspace(-4, 4, numBins)
    ]

    return qTable, bins, obsSpaceSize

if len(sys.argv) > 2:
  qTablePath = sys.argv[1]
  GAPath = sys.argv[2]
  qTable, bins, obsSpaceSize  = load_q_table(qTablePath)
else:
  qTable, bins, obsSpaceSize  = load_q_table('data/qTable_1715251781.npy')
  population = np.load('../data/GA_v2_population_100_fitness_0.01_10.npy',allow_pickle=True)


def test(qTable, env,  numEpisodes=1,  seed = SEED, GA = False, bins = bins, obsSpaceSize = obsSpaceSize):
    # Test the trained q-table with the environment
    best_reward = 0
    for episode in range(numEpisodes):
        state = env.reset(seed=seed)
        done = False
        steps = 0
        discrete_state = get_discrete_state(env.observation_space.high, bins, obsSpaceSize)
        frames = []
        while not done and steps < 500:
            frames.append(env.render())
            if GA:
                action = int(qTable[discrete_state])
            else:
                action = np.argmax(qTable[discrete_state])
            state, reward, done, _, info = env.step(action)
            discrete_state = get_discrete_state(state, bins, obsSpaceSize)
            steps += reward

        # save the frames in frames folder
        #for i, frame in enumerate(frames):
        #    plt.imsave(f'frames/RL_{episode}_{i}.png', frame)
        
        print(f'Episode {episode} reward: {steps}')
        if steps > best_reward:
             best_reward = steps

    return best_reward

def plot_results(qTable, best_ind):
    qTable = np.argmax(qTable, axis=4)
    diff = np.reshape(np.abs(qTable - best_ind),(NUM_BINS*NUM_BINS, NUM_BINS*NUM_BINS))
    # plot diff between the best individual and the qTable x: state, y: action
    plt.imshow(diff, cmap='hot', interpolation='nearest')
    plt.show()
    # save the plot
    plt.imsave('plots/diff.png',diff, cmap='hot')
    # include axis in the saved plot
    



# RL test
print('### RL Testing ###')
test(qTable, env)

# GA test
best_ind = population[0]
best_reward = 0
print('### GA Testing ###')
for i in range(len(population)):
    print(f'Testing individual {i}')
    reward = test(population[i],env, GA = True) 
    if reward > best_reward:
        best_reward = reward
        best_ind = i

print(f'Best individual {i}: {best_reward}')

plot_results(qTable, best_ind)

