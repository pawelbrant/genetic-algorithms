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
    print(parents.shape)
    for n in range(int(np.ceil(offspring_size[0]/2))):
        temp_parents = zip(parents[2*n], parents[2*n+1])
    n = 0
    # breakpoint()
    for parent_1, parent_2 in temp_parents:

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
num_individuals = 16
num_generaions = 20

# dependant
num_mating = num_individuals
pop_dim = (num_individuals, num_variables)

# creating population
x_domain=[-2,2]
y_domain=[-2,2]
#how many bits will chromosome have
precision=10
pop = np.random.randint(0, 2**precision, size=pop_dim)
pop_float=np.ndarray(shape=pop_dim , dtype=float, order='F')
for counter, individual in enumerate(pop):
    pop_float[counter,0]=individual[0]*(x_domain[1]-x_domain[0])/2**precision
    pop_float[counter,0]+=x_domain[0]
    pop_float[counter,1]=individual[1]*(y_domain[1]-y_domain[0])/2**precision
    pop_float[counter,1]+=y_domain[0]

print("Binary representation | Float representation")
for i in range(num_individuals):
    print("x = "+str(bin(pop[i,0]))+" "+str(pop_float[i,0])+" y = "+str(bin(pop[i,1]))+" "+str(pop_float[i,1]))

# running GA
for generation in range(num_generaions):
    fitness = function(pop_float)
    breakpoint()
    parents = select_parents(pop, fitness, num_mating)
    print(parents)
    pop = crossover(parents, pop_dim, crossover_prob)
    pop = mutation(pop, mutation_prob)
