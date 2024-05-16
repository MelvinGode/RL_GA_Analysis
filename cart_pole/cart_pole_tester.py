# test the trained q-table with the environment
import os
import sys
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time


N_TESTS = 10
SEED = np.random.randint(0, 1000000,N_TESTS)
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
  qTable, bins, obsSpaceSize  = load_q_table('RL/data/qTable_1_20000_10000_-375.npy')
  population = np.load('data/GA_v2_population_100_fitness_0.005_fitness_2_10.npy',allow_pickle=True)


def test(qTable, env,  numEpisodes=1,  seed = SEED[0], GA = False, bins = bins, obsSpaceSize = obsSpaceSize):
    # Test the trained q-table with the environment
    best_reward = 0
    seed = int(seed)
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

def plot_table_diff(qTable, best_ind):
    qTable = np.argmax(qTable, axis=4)
    diff = np.reshape(np.abs(qTable - best_ind),(NUM_BINS*NUM_BINS, NUM_BINS*NUM_BINS))
    # plot diff between the best individual and the qTable x: state, y: action
    plt.imshow(diff, cmap='hot', interpolation='nearest')
    plt.show()
    # save the plot
    plt.imsave('plots/diff.png',diff, cmap='hot')
    # include axis in the saved plot

def plot_results(agent_results, best_ind_rew, xlabel='Test Number', ylabel='Reward', ga_color='blue', rl_color='red'):
    
    agent_mean = np.mean(agent_results)
    agent_std_dev = np.std(agent_results)

    ga_mean = np.mean(best_ind_rew)
    ga_std_dev = np.std(best_ind_rew)

    
    plt.fill_between(range(N_TESTS+1), agent_mean - agent_std_dev, agent_mean + agent_std_dev, color=rl_color, alpha=0.05)
    plt.fill_between(range(N_TESTS+1), ga_mean - ga_std_dev, ga_mean + ga_std_dev, color=ga_color, alpha=0.05)

    plt.plot(np.arange(N_TESTS+1)[1:], best_ind_rew, label='GA', color=ga_color)
    plt.plot(np.arange(N_TESTS+1)[1:], agent_results, label='RL', color=rl_color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('plots/RL_vs_GA.png')
    plt.show()

best_ind_rew = np.zeros(N_TESTS)
agent_results = np.zeros(N_TESTS)
for i in range(N_TESTS):
    # RL test
    print('### RL Testing ###')
    agent_results[i] = test(qTable, env, seed = SEED[i])

    # GA test
    best_ind = population[0]
    print('### GA Testing ###')
    for j in range(len(population)):
        print(f'Testing individual {j}')
        reward = test(population[j],env, GA = True, seed = SEED[i]) 
        if reward > best_ind_rew[i]:
            best_ind_rew[i] = reward
            best_ind = i

    # save the best individual
    print(f'Best individual {i}: {best_ind_rew[i]}')

# plot the difference between the best individual and the qTable





plot_results(agent_results, best_ind_rew)
plot_table_diff(qTable, np.argmax(best_ind))


