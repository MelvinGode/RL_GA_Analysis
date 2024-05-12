import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from PIL import Image
from IPython.core.display import Image as img
import imageio
import shutil

env = gym.make('CartPole-v1', render_mode='rgb_array')

# How much new info will override old info. 0 means nothing is learned, 1 means only most recent is considered, old knowledge is discarded
LEARNING_RATE = 0.1
# Between 0 and 1, mesue of how much we carre about future reward over immedate reward
# default = 0.95
DISCOUNT = 1
RUNS = 2000  # Number of iterations run
SHOW_EVERY = 2000  # How oftern the current solution is rendered
UPDATE_EVERY = 100  # How oftern the current progress is recorded

# Exploration settings
epsilon = 1  # not a constant, going to be decayed (Q-LEARNING WITH EPSILON-DECAY)
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = RUNS // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# original value: -375
PUNISHMENT_FELL_OVER = -375  # Punishment for falling over

TIME_LIMIT = 500
RANDOM_SEEDS_PATH = '../data/random_seeds.npy'

# move working directory to the current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Create bins and Q table
def create_bins_and_q_table():

	numBins = 20
	obsSpaceSize = len(env.observation_space.high)

	# Get the size of each bucket
	bins = [
		np.linspace(-4.8, 4.8, numBins),
		np.linspace(-4, 4, numBins),
		np.linspace(-.418, .418, numBins),
		np.linspace(-4, 4, numBins)
	]

	qTable = np.random.uniform(low=-2, high=0, size=([numBins] * obsSpaceSize + [env.action_space.n]))

	return bins, obsSpaceSize, qTable


# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
	stateIndex = []
	for i in range(obsSpaceSize):
		stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
	return tuple(stateIndex)


# function for saving the final qtabel
def save_q_table(qTable, ts):
	np.save(f"data/qTable_{int(ts)}.npy", qTable)


# Plot graph

def plot_metrics(metrics, ts):
	# plot graph: X episodes, Y
	plt.plot(metrics['ep'], metrics['avg'], label="average rewards")
	plt.plot(metrics['ep'], metrics['min'], label="min rewards")
	plt.plot(metrics['ep'], metrics['max'], label="max rewards")
	plt.legend(loc=4)
	#plt.show()
	# save plot
	plt.savefig(f"plots/metrics_{int(ts)}.png")
	plt.clf()



def plot_time_metrics(metrics, ts):
	# plot graph: X time, Y average and max reward
	plt.plot(metrics['time'], metrics['avg'], label="average rewards")
	plt.plot(metrics['time'], metrics['min'], label="min rewards")
	plt.plot(metrics['time'], metrics['max'], label="max rewards")
	plt.title("Reward over time")
	plt.legend(loc=4)
	plt.savefig(f"plots/time_metrics_{int(ts)}.png")
	plt.clf()


def save_metrics(metrics, ts):
	np.save(f"data/metrics_{int(ts)}.npy", metrics)


bins, obsSpaceSize, qTable = create_bins_and_q_table()

previousCnt = []  # array of all scores over runs
metrics = {'ep': [], 'avg': [], 'min': [], 'max': [], 'time': []}  # metrics recorded for graph

# load random seeds
if os.path.exists(RANDOM_SEEDS_PATH):
	seeds = np.load(RANDOM_SEEDS_PATH)
else:
	seeds = np.random.randint(0, 2**32, RUNS)
	print("Random seeds generated")

average_angle = 0
# add timer to measure learning time
start_time = time.time()

for run in range(RUNS):
	observation, info = env.reset(seed=int(seeds[run]))
	discreteState = get_discrete_state(env.observation_space.high, bins, obsSpaceSize)
	done = False  # has the enviroment finished?
	cnt = 0  # how may movements cart has made

	total_run_angle = 0
	while not done and cnt < TIME_LIMIT:
		if run % SHOW_EVERY == 0:
			env.render()  # if running RL comment this out

		cnt += 1
		# Get action from Q table
		if np.random.random() > epsilon:
			action = np.argmax(qTable[discreteState])
		# Get random action
		else:
			action = np.random.randint(0, env.action_space.n)
		newState, reward, done, _, info = env.step(action)  # perform action on enviroment
		total_run_angle += abs(newState[2])

		newDiscreteState = get_discrete_state(newState, bins, obsSpaceSize)

		maxFutureQ = np.max(qTable[newDiscreteState])  # estimate of optiomal future value
		currentQ = qTable[discreteState + (action, )]  # old value

		# pole fell over / went out of bounds, negative reward
		if done and cnt < TIME_LIMIT:
			reward = PUNISHMENT_FELL_OVER

		# formula to caculate all Q values
		newQ = (1 - LEARNING_RATE) * currentQ + LEARNING_RATE * (reward + DISCOUNT * maxFutureQ)
		qTable[discreteState + (action, )] = newQ  # Update qTable with new Q value

		discreteState = newDiscreteState
	
	average_angle += total_run_angle / cnt

	# Record time metric
	if run % UPDATE_EVERY == 0:
		end_time = time.time()

	previousCnt.append(cnt)

	# Decaying is being done every run if run number is within decaying range
	if END_EPSILON_DECAYING >= run >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

	# Add new metrics for graph
	if run % UPDATE_EVERY == 0:
		latestRuns = previousCnt[-UPDATE_EVERY:]
		averageCnt = sum(latestRuns) / len(latestRuns)
		metrics['ep'].append(run)
		metrics['avg'].append(averageCnt)
		metrics['min'].append(min(latestRuns))
		metrics['max'].append(max(latestRuns))
		metrics['time'].append(end_time - start_time)
		print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))

average_angle /= RUNS

env.close()

print(f"Training finieshed with average (absolute) angle {average_angle}°")

def save_RL_gif():
	env = gym.make("CartPole-v1", render_mode="rgb_array")
	path = f'gifs/RL_{DISCOUNT}_{RUNS}_{END_EPSILON_DECAYING}_{PUNISHMENT_FELL_OVER}'

	if os.path.exists(path): shutil.rmtree(path)
	os.mkdir(path)

	observation, info = env.reset(seed = 1)
	discreteState = get_discrete_state(env.observation_space.high, bins, obsSpaceSize)
	done = False  # has the enviroment finished?
	cnt = 0  # how may movements cart has made

	while not done and cnt < TIME_LIMIT:
		# Get action from Q table
		action = np.argmax(qTable[discreteState])
		newState, reward, done, _, info = env.step(action)  # perform action on enviroment

		screen = env.render()
		im = Image.fromarray(screen).convert('RGB')
		im.save(path+f'/{cnt}.png')

		discreteState = get_discrete_state(newState, bins, obsSpaceSize)

		cnt += 1
	
	print(f"Saved {cnt} frames")


def play_RL_GIF():
	i = 0
	fnames = []
	name = f'{DISCOUNT}_{RUNS}_{END_EPSILON_DECAYING}_{PUNISHMENT_FELL_OVER}'
	while os.path.exists(f'gifs/RL_'+name+f'/{i}.png'):
		fnames.append(f'gifs/RL_'+name+f'/{i}.png')
		i+=1

	if i==0 : 
		print("Please save a GIF before trying to play it")
		return "Please save a GIF before trying to play it"

	imageio.mimsave(f'gifs/RL_'+name+f'/anim.gif', [imageio.imread(fname) for fname in fnames])
	return img(filename='gifs/RL_'+name+f'/anim.gif')



save_RL_gif()
play_RL_GIF()

"""
# Save qTable
save_q_table(qTable, start_time)
save_metrics(metrics, start_time)
# Plot metrics
plot_metrics(metrics, start_time)
plot_time_metrics(metrics, start_time)
"""