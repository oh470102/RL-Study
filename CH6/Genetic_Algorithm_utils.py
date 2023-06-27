import torch, numpy as np
import gym
from Model import Model

def unpack_parms(params, layers=[(25,4), (10,25), (2,10)]):
    unpacked_params = []
    e = 0

    for i, l in enumerate(layers):
        s, e = e, e+np.prod(l)
        weights = params[s:e].view(l)   # puts in matrix form
        s, e = e, e+l[0]
        bias = params[s:e]
        unpacked_params.extend([weights, bias])

    return unpacked_params

def spawn_population(N=50, size=407):       # N = size of population. size = number of params.
    pop = []

    for i in range(N):
        vec = torch.randn(size) / 2.0       # vec is the parameters, randomly initialized.      
        fit = 0
        p = {'params': vec, 'fitness': fit} # p is the individual. 
        pop.append(p)

    return pop

def recombine(x1, x2):                      # "shuffles" two parents
    x1 = x1['params']
    x2 = x2['params']
    l = x1.shape[0]

    split_pt = np.random.randint(l)
    child1 = torch.zeros(l)
    child2 = torch.zeros(l)

    child1[0:split_pt] = x1[0:split_pt]
    child1[split_pt:] = x2[split_pt:]
    child2[0:split_pt] = x2[0:split_pt]
    child2[split_pt:] = x1[split_pt:]

    c1 = {'params':child1, "fitness": 0.0}
    c2 = {'params':child2, "fitness": 0.0}

    return c1, c2

def mutate(x, rate=0.01):                   # randomly changes a parameter 
    x_ = x['params']
    num_to_change = int(rate * x_.shape[0])

    idx = np.random.randint(low=0, high=x_.shape[0], size=(num_to_change))
    x_[idx] = torch.randn(num_to_change) / 10.0
    
    x['params'] = x_
    return x

def test_agent(agent):
    env = gym.make('CartPole-v1')
    env.reset()

    curr_state = torch.from_numpy(env.env.state).float()
    score = 0
    done = False

    while done is False:
        params = unpack_parms(agent['params'])
        temp_agent_model = Model(params)
        probs = temp_agent_model(curr_state)
        action = torch.distributions.Categorical(probs=probs).sample()
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        curr_state = torch.from_numpy(next_state).float()
        score += 1

    return score
        
def evaluate_population(pop):
    tot_fit = 0
    lp = len(pop)

    for agent in pop:
        score = test_agent(agent)
        agent['fitness'] = score
        tot_fit += score
    
    avg_fit = tot_fit / lp
    return pop, avg_fit

def next_generation(pop, mut_rate=0.001, tournament_size=0.2):
    new_pop = []
    lp = len(pop)
    
    while len(new_pop) < len(pop):

        rids = np.random.randint(low=0, high=lp, size=(int(tournament_size*lp)))
        batch = np.array([[i, x['fitness']] for (i,x) in enumerate(pop) if i in rids]) # select a sample population subset to assure diversity
        scores = batch[batch[:, 1].argsort()]  # .argsort() returns the indices. 
        i0, i1 = int(scores[-1][0]), int(scores[-2][0]) # pick the highest two scoring agents
        parent0, parent1 = pop[i0], pop[i1]
        offspring_ = recombine(parent0, parent1)
        child1 = mutate(offspring_[0], rate=mut_rate)
        child2 = mutate(offspring_[1], rate=mut_rate)

        offspring = [child1, child2]
        new_pop.extend(offspring)
    return new_pop

def matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

