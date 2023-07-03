import torch
import numpy as np

def resolve_matplotlib_error():
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# "steals" discounted probs from neighboring supports
# in the original paper, projections are used instead.
def update_dist(r, support, probs, lim=(-10., 10.), gamma=0.8):
    nsup = probs.shape[0]
    vmin, vmax = lim[0], lim[1]
    dz = (vmax-vmin)/(nsup-1.)          # common difference
    bj = np.round( (r-vmin) / dz)       # nearest index of support corresponding to reward
    bj = int(np.clip(bj, 0, nsup-1))    # clip index
    m = probs.clone()

    j = 1
    for i in range(bj, 1, -1):          # loop through elements leftwards (descending)
        m[i] += np.power(gamma, j ) * m[i-1]
        j += 1

    j = 1
    for i in range(bj, nsup-1, 1):      # loop through elements rightwards (ascending)
        m[i] += np.power(gamma, j ) * m[i+1]
        j+= 1

    m = m / m.sum()                     # scale prob. distribution 


    return m


# state -> two F.C. layers -> action (3 possible) q-distributions
# x: input (B, 128)
# theta: parameters of the network as a vector
# aspace: size of action_space
def dist_dqn(x, theta, aspace=3):
    dim0, dim1, dim2, dim3 = 128, 100, 25, 51    # (B, 128) * (128, 100) * (100, 25) * (25, 51)
    t1 = dim0 * dim1
    t2 = dim1 * dim2
    theta1 = theta[0:t1].reshape(dim0, dim1)        # Parameter Matrix
    theta2 = theta[t1:t1 + t2].reshape(dim1, dim2)
    
    l1 = x @ theta1         # matrix multiplication
    l1 = torch.selu(l1)     
    l2 = l1 @ theta2 
    l2 = torch.selu(l2)
    l3 = []                 # not sure why, but in this chapter, NN is implemented by hands
                            # with matrix multiplication 
    for i in range(aspace): # create reward vs probability distribution for all possible actions
        step = dim2 * dim3
        theta5_dim = t1 + t2 + i * step
        theta5 = theta[theta5_dim:theta5_dim+step].reshape(dim2, dim3)
        l3_ = l2 @ theta5 
        l3.append(l3_)

    l3 = torch.stack(l3, dim=1)
    l3 = torch.nn.functional.softmax(l3, dim=2)
    return l3.squeeze()

# returns a target distribution
# updates the reward vs prob. distribution only of the action taken 
# using the "probability stealing" method 
# if last step, updated towards a degenerate distribution (p_i = 1 if i=50 else 0)
def get_target_dist(dist_batch, action_batch, reward_batch, support, lim=(-10, 10), gamma=0.8):
    nsup = support.shape[0]
    vmin, vmax = lim[0], lim[1]
    dz = (vmax-vmin)/(nsup-1.)         
    target_dist_batch = dist_batch.clone()

    for i in range(dist_batch.shape[0]): # dist_batch: B x 3 x 51
        dist_full = dist_batch[i]        # (3 x 51)
        action = int(action_batch[i].item())
        dist = dist_full[action]         # (1 x 51)
        r = reward_batch[i]

        if r != -1: # terminal step
            target_dist = torch.zeros(nsup)
            bj = np.round((r-vmin)/dz)
            bj = int(np.clip(bj, 0, nsup-1))
            target_dist[bj] = 1         # everything else has p=0

        else:
            target_dist = update_dist(r, support, dist, lim=lim, gamma=gamma)

    return target_dist_batch

# For loss function, KL Divergence is used
# to compare the "difference" between two distributions (x and y)
# x, y : B x 3 x 51
def lossfn(x, y): 
    loss = torch.Tensor([0.])
    loss.requires_grad = True

    # iterate over batches
    # and calculate Q * log(P) for each probability, and sum them
    # @ : -> inner product
    # note that this calculates loss for "all actions", including the not-chosen ones
    # such that it learns not to update unused distributions
    for i in range(x.shape[0]): 
        loss_ = -1 * torch.log(x[i].flatten(start_dim=0)) @ y[i].flatten(start_dim=0)
        loss = loss + loss_

    return loss

# np.array of 0~255 : -> torch.tensor of 0~1
def preproc_state(state):
    if isinstance(state, tuple): state = np.array(state)
    p_state = torch.from_numpy(state).unsqueeze(dim=0).float()
    p_state = torch.nn.functional.normalize(p_state, dim=1)
    return p_state

# choose action from distribution
# in this example, action with highest expectation (= reward @ prob) is chosen, but
# a more sophisticated algorithm could also consider other factors such as variance
def get_action(dist, support):
    actions = []
    # again, dist : B x 3 x 51
    for b in range(dist.shape[0]):
        expectations = [support @ dist[b, a, :] for a in range(dist.shape[1])]
        # choose argmax(E)
        action = int(np.argmax(expectations))
        actions.append(action)
    
    actions = torch.Tensor(actions).int()
    return actions