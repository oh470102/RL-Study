import gym
import numpy as np
import torch

def worker(t, worker_model, counter, params, queue): ## params: dict(epochs, n_workers)
                                              ## worker_model: MasterNode -> Actorcritic()
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    worker_opt = torch.optim.Adam(lr=1e-4, params=worker_model.parameters())
    worker_opt.zero_grad()

    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        actor_loss, critic_loss, eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value = counter.value + 1
        print(f"currently on epoch: {counter.value}/{(params['epochs']*params['n_workers'])}")
        queue.put(eplen)
        
        
def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gamma=0.95):

    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)
    returns = []
    ret_ = torch.Tensor([0])
    for r in range(rewards.shape[0]):
        ret_ = rewards[r] + gamma*ret_
        returns.append(ret_)
    returns = torch.stack(returns).view(-1)
    returns = torch.nn.functional.normalize(returns, dim=0)

    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)

    actor_loss = -1*logprobs*(returns - values.detach())
    critic_loss = torch.pow(values - returns, 2)
    loss = actor_loss.sum() + clc*critic_loss.sum()
    loss.backward()
    worker_opt.step()

    return actor_loss, critic_loss, len(rewards)

def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0

    while done is False:
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits) # logits = probability weights. Not normalized.
        action = action_dist.sample() 
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        next_state, _, done, _, _ = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(next_state).float()

        if j > 250: done = True
        if done:
            reward = -10
            worker_env.reset()
        else:
            reward = 1.0
        rewards.append(reward)

    return values, logprobs, rewards

def moving_average(array, window_size, reg=None):
    moving_averages = []
    for i in range(len(array) - window_size + 1):
        window = array[i:i+window_size]
        average = np.mean(window)
        moving_averages.append(average)

    if reg is True:
        return [i/np.max(moving_averages) for i in moving_averages]
    else:
        return moving_averages

def matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'