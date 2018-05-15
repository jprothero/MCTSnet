from wrn import WideResNet
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from ipdb import set_trace
import model_config
import config

class MCTSnet(nn.Module):
    def __init__(self, R, C, value_bottleneck=2, policy_bottleneck=2, max_sims=3, cuda=torch.cuda.is_available()):
        super(MCTSnet, self).__init__()
        self.has_cuda = cuda
        self.max_sims = max_sims
        ng = model_config.NUM_GROUPS
        k = model_config.k
        n = model_config.N

        self.num_channels = model_config.BASE_CHANNELS*model_config.NUM_GROUPS*model_config.k

        reducer = model_config.REDUCER

        self.emb_net = WideResNet(num_groups=ng, N=n, k=k, in_channels=config.CH)
        # self.exploitation_net = WideResNet(num_groups=2, N=2, k=4, in_channels=128)
        # self.exploration_net = WideResNet(num_groups=2, N=2, k=4, in_channels=128)
        self.simulation_net = WideResNet(num_groups=ng, N=n, k=k, in_channels=self.num_channels*2)
        self.value_head = nn.Sequential(*[
            WideResNet(num_groups=ng-reducer, N=n-reducer, k=k-reducer, num_classes=value_bottleneck, in_channels=self.num_channels*2),
            nn.Linear(value_bottleneck, 1),
            nn.Tanh()
        ])

        self.continue_head = nn.Sequential(*[
            WideResNet(num_groups=ng-reducer, N=n-reducer, k=k-reducer, num_classes=value_bottleneck, in_channels=self.num_channels),
            nn.Linear(value_bottleneck, 1),
            nn.Sigmoid()
        ])

        self.forget_net = WideResNet(num_groups=ng, N=n, k=k, in_channels=self.num_channels*2)
        self.policy_net = nn.Sequential(*[
            WideResNet(num_groups=ng, N=n, k=k, num_classes=policy_bottleneck, in_channels=self.num_channels*2)
            , nn.Linear(policy_bottleneck, R*C)
            ])

        self.nets = [
            self.emb_net,
            # self.exploitation_net,
            # self.exploration_net,
            self.simulation_net,
            self.value_head,
            self.continue_head,
            self.forget_net,
            self.policy_net,
        ]

        #the 128 will change if we alter the wideresnet parameters 
        assert max_sims > 2
        self.forget_bias = nn.Parameter(torch.log(torch.rand(self.num_channels, R, C)*(max_sims-2) + 1))

        self.initialize()

    def initialize(self):
        for net in self.nets:
            for _, params in net.state_dict().items():
                try:
                    nn.init.xavier_uniform_(params)
                except:
                    nn.init.uniform_(params, 0.1)

    def forward(self, state):
        embedding = F.tanh(self.emb_net(state))
        
        continues = [i for i in range(state.shape[0])]
        history = embedding.clone()
        i = 0
        #consider adding adversarial loss to get history, simulation, and embedding in the same domain
        while len(continues) > 0:
            noise = (torch.rand_like(history[continues])-.5)*2
            if self.has_cuda:
                noise = noise.cuda()
            sim_inp = torch.cat([history[continues], noise], dim=1)
            simulation = F.tanh(self.simulation_net(sim_inp))
            # simulation = F.tanh(self.exploitation_net(history) + self.exploration_net(noise) + \
            #     self.exploration_net(history))

            chance = self.continue_head(simulation)

            #sooo the issue is that we are changing history, but it cant be changed
            #because we need the original
            #
            
            new_continues = []
            chance_mul = []
            opposite_chance_mul = []
            for i, p in enumerate(chance):
                p = p.squeeze().item()
                if np.random.choice(2, p=[1-p, p]):
                    new_continues.append(i)
                chance_mul.append(p)
                opposite_chance_mul.append(1-p)
            chance_mul = torch.tensor(chance_mul).view(-1, 1, 1, 1)
            opposite_chance_mul = torch.tensor(opposite_chance_mul).view(-1, 1, 1, 1)

            if self.has_cuda:
                chance_mul = chance_mul.cuda()
                opposite_chance_mul = opposite_chance_mul.cuda()

            forget_inp = torch.cat([simulation, history[continues]], dim=1)
            forget = F.sigmoid(self.forget_net(forget_inp) + self.forget_bias)
            update = chance_mul*simulation + opposite_chance_mul*embedding[continues]
            history[continues] = history[continues]*forget + (1-forget)*update
            continues = new_continues
            i += 1
            if i > self.max_sims:
                break

        policy_val_inp = torch.cat([embedding, history], dim=1)
        combine = self.policy_net(policy_val_inp)
        # value = self.value_head(policy_val_inp)
        value = self.value_head(policy_val_inp)
        
        policy = F.softmax(combine, dim=1)

        #so what do we want to do....
        #we can try some type of GAN type thing, or we can do alpha zero
        #for alpha zero it would be that we have some base state, we run "macro" simulations
        #and we run "micro" simulation
        #I built the micro simulations, I could add a macro simulation head and try that out
        #then we probaby mix the policies based on the values, or exponentiated values

        #so again, gan loss, or alpha zero
        #we can try a modified alpha zero where we generate a prior, we do a simulation, get a new policy, update it,
        #so what would it be, basically we would do a simulation like this, then we UCT select based off of it
        #so basically we can use alpha zero training or gan training
        #I think alpha zero training is probably more ideal
        #so basically we use what we have, and when we hit a new state we run it through this net
        #and expand. then we do alpha zero like normal


        #well we could do something more complicated such as GAN loss, 
        #but for now I think we can just use this as a drop in for the alpha zero system
        #and the exciting thing about thta is that maybe I can use this for my project

        #if we're doing it like this I dont think we necessarily need the randomness,
        #because in theory we're going to be just deterministically 
        #going from node to node, unrelatedly
        #it would allow us to stochastically simulate more rather than deterministically,
        #which I guess might be good

        #we can try it with randomness for now
        
        #we can consider adding a weighting head similar to the forget gate which weighs the exploration and exploitation
        #gates, although right now it's kind of implicitly doing that.
        return policy, value


        