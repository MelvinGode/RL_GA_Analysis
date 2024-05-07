
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import os



os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""Big parameters"""

MAX_MOVES = 500
OBS_SPACE_SIZE = 4
NB_GEN = 200
POP_SIZE = 50
MUTATION_RATE = 0.005 # 0.005
NP_SEED = 10 
SELECTION = "fitness" # fitness
np.random.seed(NP_SEED)

import ga_functions as ga
"""Initial population creation"""

def create_bins(numBins=20):
    # Get the size of each bucket
	return [
		np.linspace(-4.8, 4.8, numBins),
		np.linspace(-4, 4, numBins),
		np.linspace(-.418, .418, numBins),
		np.linspace(-4, 4, numBins)
	]

# create the population using as genotype a table state:action
def create_population(pop_size, numBins=20, obsSpaceSize=4):

   # for each possible bin, create a random action
    population = np.empty((pop_size, *([numBins] * obsSpaceSize)))
    for i in range(pop_size):
        population[i] = np.random.choice([0,1], ([numBins] * obsSpaceSize))
    return population

# Given a state of the enviroment, return its descreteState index in qTable
def get_discrete_state(state, bins, obsSpaceSize):
	stateIndex = []
	for i in range(obsSpaceSize):
		stateIndex.append(np.digitize(state[i], bins[i]) - 1) # -1 will turn bin into index
	return tuple(stateIndex)


"""###Environment functions"""

def play_gen(population, env, random_seed_array, bins, obsSpaceSize=OBS_SPACE_SIZE, max_moves=MAX_MOVES):
  pop_size = population.shape[0]
  fitnesses = np.empty(pop_size)
  for i in range(pop_size):
    env.reset(seed=int(random_seed_array[i]))
    discreteState = get_discrete_state(env.observation_space.high, bins, obsSpaceSize)
    #print(f'Individual {i}, play gen...')
    for t in range(max_moves):
      
      action = population[i][discreteState]
      observation, reward, done, info, blc = env.step(action)
      discreteState = get_discrete_state(observation, bins, obsSpaceSize)
      if done : break

    fitnesses[i]=t
    #print(f'Individual {i}, Fitness: {fitnesses[i]}')

  return fitnesses

"""###Environment creation"""

env = gym.make("CartPole-v1")

class GA_agent():

    def __init__(self, selection, nb_gen, pop_size, mutation_rate, elitism=0):
        if selection == "rank" : self.select = ga.rank_selection
        elif selection=="fitness" : self.select = ga.fitness_selection
        else : return "Unknown selection method"

        self.nb_gen = nb_gen
        self.pop_size = pop_size
        self.numBins = 20
        self.bins = create_bins(self.numBins)
        self.population = create_population(pop_size)
        print(self.population.shape)
        self.mutation_rate = mutation_rate
        self.elitism = elitism

        self.best_score = 0

        self.best_scores = np.empty(nb_gen)
        self.avg_scores = np.empty(nb_gen)
        self.diversity = np.empty(nb_gen)
        self.time_samples = np.empty(nb_gen)

        # if exists the file data/random_seed.npy, load it
        if os.path.exists('../data/random_seed.npy'):
            self.random_seed_array = np.load('../data/random_seed.npy')
        else:
            self.random_seed_array = np.random.randint(0, 2**32, 10000)
        self.current_random_seed = 0

    def evolve(self, env = gym.make("CartPole-v1")):

        last_time = time.time()
        self.start_time  = last_time
        start_time = last_time

        for i in range(self.nb_gen):

            self.population = self.population.astype(int)

            self.diversity[i] = ga.pop_diversity(self.population)
            print(f'### Generation {i}: starting playing... ###')
            fit = play_gen(self.population, env, self.random_seed_array[self.current_random_seed: self.current_random_seed + self.pop_size],  bins = self.bins)
            self.current_random_seed += self.pop_size

            if max(fit) > self.best_score: self.best_score = max(fit)

            now = time.time()
            self.time_samples[i] = now - start_time

            self.elite = self.population[np.argmax(fit)].copy()

            self.best_scores[i] = max(fit)
            self.avg_scores[i] = np.mean(fit)

            # Selection
            pairs = self.select(fit)
            # Crossover
            self.population = ga.crossover(pairs, self.population)
            # Mutation
            self.population = ga.mutation(self.population, self.mutation_rate, numBins=self.numBins)
            # Elitism
            self.population[np.random.choice(range(self.pop_size))] = self.elite

        env.close()

    def plot_evolution(self):
        plt.plot(self.best_scores, label="Best score")
        plt.plot(self.avg_scores, label="Average score")
        plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness over time")
        plt.savefig(f'plots/GA_v2_evolution_{SELECTION}_{self.mutation_rate}_{NP_SEED}.png')
        plt.show()

    def plot_evolution_per_second(self):
        plt.plot(self.time_samples, self.best_scores, label="Max scores")
        plt.plot(self.time_samples, self.avg_scores, label="Mean score")
        plt.legend()
        plt.xlabel("Seconds of training")
        plt.ylabel("Fitness")
        plt.title("Fitness over training time")
        plt.savefig(f'plots/GA_v2_evoluiton_per_second_{SELECTION}_{self.mutation_rate}_{NP_SEED}.png')
        plt.show()

    def plot_diversity(self):
        plt.plot(self.diversity, color="orange")
        plt.xlabel("Generation")
        plt.ylabel("Average gene STD")
        plt.title("Population diversity over time")
        plt.savefig(f'plots/GA_v2_pop_diversity_{SELECTION}_{self.mutation_rate}_{NP_SEED}.png')
        plt.show()

    def save_metrics(self):
        name = f'data/GA_v2_metrics_{SELECTION}_{self.mutation_rate}_{NP_SEED}.npy'
        metrics = np.array({
            "max" : self.best_scores,
            "avg" : self.avg_scores,
            "diversity" : self.diversity,
            "time": self.time_samples})
        np.save(name, metrics)

    def savePopulation(self):
        name = f'../data/GA_v2_population_{SELECTION}_{self.mutation_rate}_{NP_SEED}.npy'
        np.save(name, self.population)


agent = GA_agent(SELECTION, NB_GEN, POP_SIZE, MUTATION_RATE)

agent.evolve()

agent.plot_evolution_per_second()
agent.plot_evolution()
agent.plot_diversity()
agent.savePopulation()
agent.save_metrics()
