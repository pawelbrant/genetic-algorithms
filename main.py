import random

import numpy as np
import matplotlib.pyplot as plt


def function(population):
    '''mathematic function given in excercise'''
    list = [((np.square(x) - np.square(y)) - np.square(1 - x)) for x, y in population]
    value = np.asarray(list)
    return value

def select_parents(pop, fitness, num_parents):
    '''randomly selects given number of parents from population weighted by fitness'''
    parents = np.empty((num_parents, pop.shape[1]))
    fitness_exp = np.exp(fitness)
    for n in range(num_parents):
        rnd = random.random() * sum(fitness_exp)
        for i, w in enumerate(fitness_exp):
            rnd -= w
            if rnd < 0:
                parents[n] = pop[i]
                break
    return parents

def crossover(parents, offspring_size, prob):
    offspring = np.empty(offspring_size)
    n = 0
    for parent_1 in parents:
        for parent_2 in parents:
            if n >= offspring_size[0]:
                return offspring
            rnd = random.random()
            if rnd < prob:
                offspring[n, 0] = parent_1[0]
                offspring[n, 1] = parent_2[1]
                n += 1
                if n >= offspring_size[0]:
                    return offspring
                offspring[n, 0] = parent_2[0]
                offspring[n, 1] = parent_1[1]
                n += 1
            else:
                offspring[n] = parent_1
                n += 1
                if n >= offspring_size[0]:
                    return offspring
                offspring[n] = parent_2
                n += 1

def mutation(population, prob):
    mutated = np.empty(population.shape)
    for n, individual in enumerate(population):
        rnd = random.random()
        if rnd < prob:
            mutated[n, 0] = float(np.invert(individual[0].astype(np.int)))
            mutated[n, 1] = float(np.invert(individual[1].astype(np.int)))
        else:
            mutated[n] = individual
    return mutated

# non-changing variable
num_variables = 2

#changing variables
crossover_prob = 1
mutation_prob = 0.7
num_individuals = 15
num_generaions = 20

# dependant
num_mating = num_individuals
pop_dim = (num_individuals, num_variables)

# creating population
pop = np.random.uniform(low=-2.0, high=2.0, size=pop_dim)
print(pop)

# running GA
for generation in range(num_generaions):
    fitness = function(pop)
    parents = select_parents(pop, fitness, num_mating)
    print(parents)
    pop = crossover(parents, pop_dim, crossover_prob)
    pop = mutation(pop, mutation_prob)

# breakpoint()
