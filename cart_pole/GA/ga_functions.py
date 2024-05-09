import numpy as np

def fitness_selection(fitnesses):
  half_pop_size = int(len(fitnesses)/2)
  pairs = np.empty((half_pop_size,2))

  selection_probabilities = fitnesses/sum(fitnesses)

  for i in range(half_pop_size):
    pairs[i] = np.random.choice(range(len(fitnesses)), size=2, p=selection_probabilities, replace=False)
  return pairs.astype(int)

def rank_selection(fitnesses):
    half_pop_size = int(len(fitnesses)/2)
    pairs = np.empty((half_pop_size,2))

    selection_probabilities = np.argsort(fitnesses)/sum(range(len(fitnesses)))

    for i in range(half_pop_size):
      pairs[i] = np.random.choice(range(len(fitnesses)), size=2, p=selection_probabilities, replace=False)
    return pairs.astype(int)

def crossover(pairs, population, numBins=20, obsSpaceSize=4):
  pop_size = population.shape[0]

  new_gen = np.empty((pop_size, *([numBins] * obsSpaceSize)))
  for pair_nb in range(pairs.shape[0]):
    pair = pairs[pair_nb]
    dad = population[pair[0],:]
    mom = population[pair[1],:]

    for i in range(len(dad)):
      if np.random.choice([True,False]):
        new_gen[2*pair_nb, i] = dad[i]
        new_gen[2*pair_nb+1, i] = mom[i]
      else:
        new_gen[2*pair_nb, i] = mom[i]
        new_gen[2*pair_nb+1, i] = dad[i]

  return new_gen

def one_point_crossover(pairs, population, crossoverpoint=8000,numBins=20, obsSpaceSize=4):
  pop_size = population.shape[0]

  new_gen = np.empty((pop_size, *([numBins] * obsSpaceSize)))
  for pair_nb in range(pairs.shape[0]):
    pair = pairs[pair_nb]
    dad = population[pair[0],:]
    mom = population[pair[1],:]

    for i in range(len(dad)):
      if i < crossoverpoint:
        new_gen[2*pair_nb, i] = dad[i]
        new_gen[2*pair_nb+1, i] = mom[i]
      else:
        new_gen[2*pair_nb, i] = mom[i]
        new_gen[2*pair_nb+1, i] = dad[i]


def mutation(population, mutation_rate, numBins=20):
  pop_size = population.shape[0]

  for i in range(pop_size):
    for j in range(len(population[i])):
       if np.random.rand() <= mutation_rate:
         population[i][j] = np.random.randint(0,1)
       
  return population

def pop_diversity(population):
  return np.std(population, 0).mean()

