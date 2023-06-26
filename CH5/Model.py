import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_layer = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_layer = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=0)
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        ## ACTOR SIDE
        actor = F.log_softmax(self.actor_layer(out), dim=0)

        ## CRITIC SIDE - 역전파 수행 X
        critic = F.relu(self.l3(out.detach()))
        critic = torch.tanh(self.critic_layer(critic))
        
        return actor, critic