import numpy as np
import math
import random
import matplotlib.pyplot as plt


class SMO:
    def __init__(self, pop_size, dim, global_leader_limit, local_leader_limit, n_iter):
        self.population_size = pop_size
        self.SM = np.zeros((pop_size, dim))
        self.function_values = np.zeros(pop_size)
        self.probabilities = np.zeros(pop_size)
        self.fitness = np.zeros(pop_size)
        self.dimension = dim
        self.local_leader_limit = local_leader_limit
        self.local_leader_count = np.zeros(1)
        self.global_leader_limit = global_leader_limit
        self.global_leader_count = 0
        self.global_leader = np.zeros(dim)
        self.local_leader = np.zeros((1,dim))
        self.groups = [[0, pop_size]]
        self.n_groups = 1
        self.n_iter = n_iter
        self.max_n_groups = 5

        self.pr = 0.3


    def CalculateFitness(self,fun1):
        if fun1 >= 0:
            result = (1/(fun1+1))
        else:
            result=(1+math.fabs(fun1))
        return result

    def calc_fitness(self):
        for i in range(self.population_size):
            self.function_values[i] = self.function(self.SM[i])
            self.fitness[i] = self.CalculateFitness(self.function_values[i])

    def initialize(self):


        for i in range(self.population_size):
            for j in range(self.dimension):
                self.SM[i][j] = self.SM_min[j] + np.random.uniform(0, 1) * (self.SM_max[j] - self.SM_min[j])

        self.calc_fitness()

        self.global_leader_learning_phase()
        self.local_leader_learning_phase()


        # ploting
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x = self.SM[:,0]
        self.y = self.SM[:,1]
        colors = np.random.rand(self.population_size)
        self.sc = self.ax.scatter(self.x,self.y, c=colors)

        plt.xlim(self.SM_min[0],self.SM_max[0])
        plt.ylim(self.SM_min[1],self.SM_max[1])

    def global_leader_learning_phase(self):
        best_pos = np.argmin(self.fitness, axis=0)
        if (self.global_leader == self.SM[best_pos]).all():
            self.global_leader_count = self.global_leader_count + 1
        else:
            self.global_leader_count = 0

        self.global_leader = self.SM[best_pos].copy()


    def local_leader_learning_phase(self):
        for k in range(self.n_groups):
            best_local_pos = np.argmin(self.fitness[self.groups[k][0]: self.groups[k][1]])

            if (self.local_leader[k] == self.SM[self.groups[k][0]: self.groups[k][1]][best_local_pos]).all():
                self.local_leader_count[k] = self.local_leader_count[k] + 1
            else:
                self.local_leader_count[k] = 0
            self.local_leader[k] = self.SM[self.groups[k][0]: self.groups[k][1]][best_local_pos].copy()


    def local_leader_phase(self):
        for k in range(self.n_groups):
            group_b = self.groups[k][0]
            group_e = self.groups[k][1]
            for i in range(group_b, group_e):
                new_SM = np.zeros(self.dimension)
                for j in range(self.dimension):
                    if np.random.uniform(0, 1) >= self.pr:
                        while True:
                            r=int(random.random()*(group_e-group_b)+group_b)
                            if (r != i):
                                break
                        new_SM[j] = self.SM[i][j] + np.random.uniform(0, 1) * (self.local_leader[k][j] - self.SM[i][j]) + np.random.uniform(-1, 1) * (self.SM[r][j] - self.SM[i][j])
                    else:
                        new_SM[j] = self.SM[i][j]

                if self.CalculateFitness(self.function(new_SM)) > self.fitness[i]:
                    self.SM[i] = new_SM
                    self.function_values[i] = self.function(new_SM)
                    self.fitness[i] = self.CalculateFitness(self.function(new_SM))


    def global_leader_phase(self):
        self.calculate_probabibilities()
        for k in range(self.n_groups):
            group_b = self.groups[k][0]
            group_e = self.groups[k][1]
            for i in range(group_b, group_e):
                if np.random.uniform(0, 1) < self.probabilities[i]:
                    new_SM = self.SM[i].copy()
                    j = int(random.random() * self.dimension)
                    while True:
                        r=int(random.random()*(group_e-group_b)+group_b)
                        if (r != i):
                            break
                    new_SM[j] = self.SM[i][j] + np.random.uniform(0, 1) * (self.global_leader[j] - self.SM[i][j]) + np.random.uniform(-1, 1) * (self.SM[r][j] - self.SM[i][j])
                    if self.CalculateFitness(self.function(new_SM)) > self.fitness[i]:
                        self.SM[i] = new_SM
                        self.function_values[i] = self.function(new_SM)
                        self.fitness[i] = self.CalculateFitness(self.function(new_SM))

    def local_leader_decision_phase(self):
        for k in range(self.n_groups):
            group_b = self.groups[k][0]
            group_e = self.groups[k][1]
            if self.local_leader_count[k] > self.local_leader_limit:
                self.local_leader_count[k] = 0
                for i in range(group_b, group_e):
                    for j in range(0, self.dimension):
                        if np.random.uniform(0, 1) >= self.pr:
                            self.SM[i][j] = self.SM_min[j] + np.random.uniform(0, 1) * (self.SM_max[j] - self.SM_min[j])
                        else:
                            self.SM[i][j] = self.SM[i][j] + np.random.uniform(0, 1) * (self.global_leader[j] - self.SM[i][j]) + np.random.uniform(0, 1) * (self.SM[i][j] - self.local_leader[k][j])

    def global_leader_decision_phase(self):
        if self.global_leader_count > self.global_leader_limit:
            self.global_leader_count = 0
            if self.n_groups < self.max_n_groups:
                self.n_groups = self.n_groups + 1
            else:
                self.n_groups = 1
            self.create_groups()
            self.local_leader_learning_phase()

    def create_groups(self):
        self.groups = []
        group_size = self.population_size / self.n_groups
        for i in range(0, self.n_groups):
            self.groups = self.groups + [[int(i * group_size), int((i + 1) * group_size)]]
        self.groups[-1][1] = self.population_size
        self.local_leader_count = np.zeros(self.n_groups)
        self.local_leader = np.zeros((self.n_groups, self.dimension))


    def set_function(self, fun, lim_min, lim_max):
        self.function = fun
        self.SM_min = lim_min
        self.SM_max = lim_max

    def optimize(self):
        for it in range(0, self.n_iter):
            print("iteracion: ", it)
            self.state()
            self.local_leader_phase()
            self.global_leader_phase()
            self.fix_limits()
            self.global_leader_learning_phase()
            self.local_leader_learning_phase()
            self.local_leader_decision_phase()
            self.global_leader_decision_phase()

    def calculate_probabibilities(self):
        for k in range(self.n_groups):
            group_b = self.groups[k][0]
            group_e = self.groups[k][1]
            max_fitness = self.fitness[group_b:group_e].max()
            for i in range(group_b, group_e):
                self.probabilities[i] = 0.9 * self.fitness[i] / max_fitness + 0.1


    def fix_limits(self):
        for i in range(self.population_size):
            for j in range(self.dimension):
                self.SM[i][j] = np.clip(self.SM[i][j], self.SM_min[j], self.SM_max[j])

    def state(self):

        # self.x = self.SM[:,0]
        # self.y = self.SM[:,1]
        # self.sc.set_offsets(np.c_[self.x,self.y])
        # plt.pause(0.1)
        # self.fig.canvas.draw()

        print("")
        print("---------------------------------------------")
        print("Best global: ", self.global_leader, "  -->  ", self.function(self.global_leader))
        print("Global count: ", self.global_leader_count)
        print("")
        print("Bests local: \n", self.local_leader)
        print("Local counts: \n", self.local_leader_count)
        print("")
        print("SM: \n", self.SM)
        print("---------------------------------------------")
        print("")
