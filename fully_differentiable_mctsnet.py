from wrn import WideResNet
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch

class MCTSnet(nn.Module):
    def __init__(self, num_choices, value_bottleneck=2, policy_bottleneck=2, max_sims=10):
        super(MCTSnet, self).__init__()
        self.max_sims = max_sims

        self.emb_net = WideResNet(num_groups=2, N=2, k=4)
        self.simulation_net = WideResNet(num_groups=2, N=2, k=4)
        self.value_head = nn.Sequential(*[
            WideResNet(num_groups=1, N=1, k=2, num_classes=value_bottleneck),
            self.flatten,
            nn.Linear(value_bottleneck, 1),
            nn.Tanh()
        ])

        self.continue_head = nn.Sequential(*[
            WideResNet(num_groups=1, N=1, k=2, num_classes=value_bottleneck),
            self.flatten, 
            nn.Linear(value_bottleneck, 1),
            nn.Sigmoid()
        ])

        self.forget_net = WideResNet(num_groups=2, N=2, k=4)
        self.backup_net = WideResNet(num_groups=2, N=2, k=4)
        self.readout_net = nn.Sequential(*[
            WideResNet(num_groups=2, N=2, k=4, num_classes=policy_bottleneck)
            , self.flatten
            , nn.Linear(policy_bottleneck, num_choices)
            ])

        self.nets = [
            self.emb_net,
            self.simulation_net,
            self.value_head,
            self.continue_head,
            self.forget_net,
            self.backup_net,
            self.readout_net,
        ]

        self.forget_bias = nn.Parameter(torch.log(torch.rand(128)*(max_sims-2) + 1))

        self.initialize()

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def initialize(self):
        for net in self.nets:
            for name, params in net.state_dict().items():
                if "weight" in name:
                    nn.init.xavier_uniform_(params)
                else:
                    nn.init.constant_(params, 0.1)

    def forward(self, state):
        values = []
        embedding = F.tanh(self.emb_net(state))
        
        history = embedding
        cont = True
        i = 0
        while cont:
            simulation = self.simulation_net(history)
            values.append(self.value_head(simulation))
            chance = self.continue_head(simulation)
            cont = np.random.choice(2, p=[1-chance, chance])
            forget = F.softmax(self.forget_net(simulation) + self.forget_net(history) + self.forget_bias, dim=1)
            update = chance*simulation + (1 - chance)*embedding
            history = history*forget + (1-forget)*update
            i += 1
            if i > self.max_sims:
                break
        
        probas = F.softmax(self.readout_net(embedding) + self.readout_net(history), dim=1)

        return probas


        