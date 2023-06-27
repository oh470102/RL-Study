### Solving CartPole with evolutionary RL: genetic algorithm.
### In genetic algorithms, the model's parameters are not "learned." That is,
### No backpropagation or SGD occurs. 

from Model import Model
from Genetic_Algorithm_utils import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym

### MATPLOTLIB ERROR RESOLVING
matplotlib_error()

### HYPERPARAMS
num_generations = 20
population_size = 500
mutation_rate = 0.01

### MODEL
pop_fit = []
pop = spawn_population(N=population_size, size=407)

def train_model():
    global pop

    for i in range(num_generations):
        print(f"currently on {i+1}/{num_generations} generation")
        pop, avg_fit = evaluate_population(pop)
        pop_fit.append(avg_fit)
        pop = next_generation(pop, mut_rate=mutation_rate, tournament_size=0.2)

train_model()
print("training complete")
plt.plot(pop_fit)
plt.show()


