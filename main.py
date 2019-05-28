import random

import numpy as np
import matplotlib.pyplot as plt

def function(population):
    '''mathematic function given in excercise'''
    list = [((np.square(x) - np.square(y)) - np.square(1 - x)) for x, y in population]
    value = np.asarray(list)
    return value

def select_parents(pop, fitness):
    '''randomly selects given number of parents from population weighted by fitness'''
    parents = np.zeros(pop.shape, dtype=object)
    fitness_exp = np.exp(fitness)
    for n in range(pop.shape[0]):
        rnd = random.random() * sum(fitness_exp)
        for i, w in enumerate(fitness_exp):
            rnd -= w
            if rnd < 0:
                parents[n,0] = format(pop[i,0], '#0'+str(precision+2)+'b')
                parents[n,1] = format(pop[i,1], '#0'+str(precision+2)+'b')
                break
    return parents

def crossover(parents, offspring_size, prob):
    offspring = np.zeros(offspring_size, dtype=object)
    n = 0
    for k in range(offspring_size[0]):
        if n >= offspring_size[0]:
            return offspring
        rnd = random.random()
        if rnd < prob:
            pivot = random.randint(3, len(parents[1,1])-1)
            offspring[n, 0] = parents[n, 0][:pivot] + parents[n+1, 1][pivot:]
            offspring[n, 1] = parents[n+1, 1][:pivot] + parents[n, 0][pivot:]
            n += 1
            if n >= offspring_size[0]:
                return offspring
            offspring[n, 0] = parents[n, 0][:pivot] + parents[n-1, 1][pivot:]
            offspring[n, 1] = parents[n-1, 1][:pivot] + parents[n, 0][pivot:]
            n += 1
        else:
            offspring[n] = parents[n]
            n += 1
            if n >= offspring_size[0]:
                return offspring
            offspring[n] = parents[n]
            n += 1
    return offspring

def mutation(population, prob):
    mutated = np.zeros(population.shape, dtype=object)
    for n, individual in enumerate(population):
        rnd = random.random()
        mutated[n] = individual
        if rnd < prob:
            pivot = random.randint(2, len(parents[1,1])-1)
            print(type(mutated[n, 0]))
            if mutated[n, 0][pivot] == '0':
                mutated[n, 0] = mutated[n, 0][:pivot] + '1' + mutated[n, 0][pivot+1:]
            else:
                mutated[n, 0] = mutated[n, 0][:pivot] + '0' + mutated[n, 0][pivot+1:]
            if mutated[n, 1][pivot] == '0':
                mutated[n, 1] = mutated[n, 1][:pivot] + '0' + mutated[n, 1][pivot+1:]
            else:
                mutated[n, 1] = mutated[n, 1][:pivot] + '1' + mutated[n, 1][pivot+1:]
    return mutated

def relaxation_function(pop, pop_dim):
    pop_float=np.ndarray(shape=pop_dim , dtype=float, order='F')
    for counter, individual in enumerate(pop):
        pop_float[counter,0]=individual[0]*(x_domain[1]-x_domain[0])/2**precision
        pop_float[counter,0]+=x_domain[0]
        pop_float[counter,1]=individual[1]*(y_domain[1]-y_domain[0])/2**precision
        pop_float[counter,1]+=y_domain[0]
    return pop_float

def bin2int(pop):
    new_population = np.zeros(pop.shape, dtype=object)
    for i, new_individual in enumerate(new_population):
        new_individual[0]=int(pop[i, 0], 2)
        new_individual[1]=int(pop[i, 1], 2)
    return new_population

# non-changing variable
num_variables = 2

#changing variables
crossover_prob = 0.6
mutation_prob = 0.2
num_individuals = 16
num_generations = 100

# dependant
pop_dim = (num_individuals, num_variables)

# creating population
x_domain=[-2,2]
y_domain=[-2,2]
#how many bits will chromosome have
precision=10
pop = np.random.randint(0, 2**precision, size=pop_dim)
pop_float=relaxation_function(pop,pop_dim)

print("Binary representation | Float representation")
for i in range(num_individuals):
    print("x = "+str(format(pop[i,0], '#0'+str(precision+2)+'b'))+" "+str(pop_float[i,0])+" y = "+str(format(pop[i,1], '#0'+str(precision+2)+'b'))+" "+str(pop_float[i,1]))

best_solution_in_each_generation=[]
#function to be optimized
# plt.plot()

# running GA
for generation_index, generation in enumerate(range(num_generations)):
    pop_float=relaxation_function(pop,pop_dim)
    plt.plot(pop_float[:,0],pop_float[:,1],'bo', alpha=0.5,label="Obiekty w populacji")
    plt.title("Wykres f(x,y). Pokolenie: "+str(generation_index+1))
    plt.ylim(1.2*y_domain[0],1.2*y_domain[1])
    plt.xlim(1.2*x_domain[0],1.2*x_domain[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
    fitness = function(pop_float)
    best_solution_in_each_generation.append(max(fitness))
    parents = select_parents(pop, fitness)
    print(parents)
    pop = crossover(parents, pop_dim, crossover_prob)
    pop = mutation(pop, mutation_prob)
    pop = bin2int(pop)

#PLOTS
plt.plot(range(1,num_generations+1),best_solution_in_each_generation,label="Best solution")
plt.ylabel("Wartość funkcji dopasowania")
plt.xlabel("Generacja")
plt.title("Wykres wartości najlepszego przystosowania osiągniętego w każdej iteracji")
if num_generations<20:
    plt.xticks(range(1,num_generations+1))

z1=np.polyfit(range(1,num_generations+1),best_solution_in_each_generation,7)
p_1=np.poly1d(z1)
xp=np.linspace(start=1.0,stop=float(num_generations+2), num=100)
p11 = plt.plot(xp,p_1(xp),'b:',alpha=0.3,label="Krzywa dopasowania")
plt.legend()
plt.grid()
plt.show()
