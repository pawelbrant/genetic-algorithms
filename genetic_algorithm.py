import random

import numpy as np
import matplotlib.pyplot as plt
import function_parser as fp


class GA:

    def __init__(self, gui_x_domain, gui_y_domain, gui_precision=10, gui_crossover_prob=0.6, gui_mutation_prob=0.1,
                 gui_num_ind=15, gui_num_gen=20):

        # non-changing variable
        self.num_variables = 2

        # changing variables
        self.crossover_prob = gui_crossover_prob
        self.mutation_prob = gui_mutation_prob
        self.num_individuals = gui_num_ind
        self.num_generations = gui_num_gen

        # dependant
        self.pop_dim = (self.num_individuals, self.num_variables)

        # creating population
        self.x_domain = gui_x_domain
        self.y_domain = gui_y_domain

        # how many bits will chromosome have
        self.precision = gui_precision
        self.pop = np.random.randint(0, 2 ** self.precision, size=self.pop_dim)
        self.pop_float = self.relaxation_function()

        self.best_solution_in_each_generation = []
        self.mean_solution_in_each_generation = []
        self.median_solution_in_each_generation = []

    def select_parents(self, fitness):
        """randomly selects given number of parents from population weighted by fitness"""
        parents = np.zeros(self.pop.shape, dtype=object)
        fitness_exp = np.exp(fitness)
        for n in range(self.pop.shape[0]):
            rnd = random.random() * sum(fitness_exp)
            for i, w in enumerate(fitness_exp):
                rnd -= w
                if rnd < 0:
                    parents[n, 0] = format(self.pop[i, 0], '#0' + str(self.precision + 2) + 'b')
                    parents[n, 1] = format(self.pop[i, 1], '#0' + str(self.precision + 2) + 'b')
                    break
        return parents

    def crossover(self, parents):
        offspring = np.zeros(self.pop_dim, dtype=object)
        n = 0
        for k in range(self.pop_dim[0]):
            if n >= self.pop_dim[0]:
                return offspring
            rnd = random.random()
            if rnd < self.crossover_prob:
                pivot = random.randint(3, self.precision - 1)
                offspring[n, 0] = parents[n, 0][:pivot] + parents[n + 1, 1][pivot:]
                offspring[n, 1] = parents[n + 1, 1][:pivot] + parents[n, 0][pivot:]
                n += 1
                if n >= self.pop_dim[0]:
                    return offspring
                offspring[n, 0] = parents[n, 0][:pivot] + parents[n - 1, 1][pivot:]
                offspring[n, 1] = parents[n - 1, 1][:pivot] + parents[n, 0][pivot:]
                n += 1
            else:
                offspring[n] = parents[n]
                n += 1
                if n >= self.pop_dim[0]:
                    return offspring
                offspring[n] = parents[n]
                n += 1
        return offspring

    def mutation(self):
        mutated = np.zeros(self.pop.shape, dtype=object)
        for n, individual in enumerate(self.pop):
            rnd = random.random()
            mutated[n] = individual
            if rnd < self.mutation_prob:
                pivot = random.randint(2, self.precision - 1)
                if mutated[n, 0][pivot] == '0':
                    mutated[n, 0] = mutated[n, 0][:pivot] + '1' + mutated[n, 0][pivot + 1:]
                else:
                    mutated[n, 0] = mutated[n, 0][:pivot] + '0' + mutated[n, 0][pivot + 1:]
                if mutated[n, 1][pivot] == '0':
                    mutated[n, 1] = mutated[n, 1][:pivot] + '0' + mutated[n, 1][pivot + 1:]
                else:
                    mutated[n, 1] = mutated[n, 1][:pivot] + '1' + mutated[n, 1][pivot + 1:]
        return mutated

    def relaxation_function(self):
        pop_float = np.ndarray(shape=self.pop_dim, dtype=float, order='F')
        for counter, individual in enumerate(self.pop):
            pop_float[counter, 0] = individual[0] * (self.x_domain[1] - self.x_domain[0]) / 2 ** self.precision
            pop_float[counter, 0] += self.x_domain[0]
            pop_float[counter, 1] = individual[1] * (self.y_domain[1] - self.y_domain[0]) / 2 ** self.precision
            pop_float[counter, 1] += self.y_domain[0]
        return pop_float

    def bin2int(self):
        new_population = np.zeros(self.pop.shape, dtype=object)
        for i, new_individual in enumerate(new_population):
            new_individual[0] = int(self.pop[i, 0], 2)
            new_individual[1] = int(self.pop[i, 1], 2)
        return new_population

    def print_bin_float_representation(self):
        for i in range(self.num_individuals):
            print("x = " + str(format(self.pop[i, 0], '#0' + str(self.precision + 2) + 'b')) + " "
                  + str(self.pop_float[i, 0]) +
                  " y = " + str(format(self.pop[i, 1], '#0' + str(self.precision + 2) + 'b')) + " "
                  + str(self.pop_float[i, 1]))


if __name__ == "__main__":
    g = GA(
        gui_x_domain=[0, 1],
        gui_y_domain=[0, 3.14],
        gui_precision=10,
        gui_crossover_prob=0.6,
        gui_mutation_prob=0.1,
        gui_num_ind=20,
        gui_num_gen=15,
    )
    function = "x**2+sin(y)"
    best_x = 0
    best_y = 0
    best_solution = 0
    # running GA
    for generation_index, generation in enumerate(range(g.num_generations)):
        plt.plot(g.pop_float[:, 0], g.pop_float[:, 1], 'bo', alpha=0.5, label="Obiekty w populacji")
        plt.title("Wykres f(x,y). Pokolenie: " + str(generation_index + 1))
        plt.ylim(1.2 * g.y_domain[0], 1.2 * g.y_domain[1])
        plt.xlim(1.2 * g.x_domain[0], 1.2 * g.x_domain[1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()
        fitness = fp.fitness_function(function, g.pop_float)
        for fit_index, fit in enumerate(fitness):
            if fit == max(fitness):
                best_x = g.pop_float[fit_index, 0]
                best_y = g.pop_float[fit_index, 1]
                best_solution = fit

        g.best_solution_in_each_generation.append(np.max(fitness))
        g.mean_solution_in_each_generation.append(np.mean(fitness))
        g.median_solution_in_each_generation.append(np.median(fitness))
        parents = g.select_parents(fitness)
        g.pop = g.crossover(parents)
        g.pop = g.mutation()
        g.pop = g.bin2int()
        g.pop_float = g.relaxation_function()

    # PLOTS
    p1 = plt.plot(range(1, g.num_generations + 1), g.best_solution_in_each_generation, color="cyan", linestyle='-',marker="x", alpha=0.5, label="Best")
    p2 = plt.plot(range(1, g.num_generations + 1), g.mean_solution_in_each_generation, color="aquamarine", linestyle='-', marker="s", alpha=0.5, label="Mean")
    p3 = plt.plot(range(1, g.num_generations + 1), g.median_solution_in_each_generation, color="teal", linestyle='-', marker="p", alpha=0.5, label="Median")
    plt.ylabel("Wartość funkcji dopasowania")
    plt.xlabel("Generacja")
    plt.title("Wykres wartości osiągniętych w każdej iteracji")
    if g.num_generations < 20:
        plt.xticks(range(1, g.num_generations + 1))
    plt.legend()
    plt.grid()
    plt.show()
    print("Best solution: "+str(best_solution)+" x: "+str(best_x)+" y: "+str(best_y))
