from wrn import WideResNet
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch
from ipdb import set_trace
import model_config
import config

class MCTSnet(nn.Module):
    def __init__(self, R, C, value_bottleneck=2, policy_bottleneck=2, continue_bottleneck = 2, cuda=torch.cuda.is_available()):
        super(MCTSnet, self).__init__()
        self.has_cuda = cuda

        ng = model_config.NUM_GROUPS
        k = model_config.MULTIPLIER
        n = model_config.NUM_LAYERS

        small_ng = model_config.SMALL_NG
        small_k = model_config.SMALL_K
        small_n = model_config.SMALL_N

        self.num_channels = model_config.BASE_CHANNELS*ng*k

        self.embed = WideResNet(num_groups=ng, N=n, k=k, in_channels=config.CH)

        self.value_head = nn.Sequential(*[
            WideResNet(num_groups=small_ng, N=small_n, k=small_k, num_classes=value_bottleneck, in_channels=self.num_channels),
            nn.Linear(value_bottleneck, 1),
            nn.Tanh()
        ])

        self.policy_head = nn.Sequential(*[
            WideResNet(num_groups=small_ng, N=small_n, k=small_k, num_classes=policy_bottleneck, in_channels=self.num_channels)
            , nn.Linear(policy_bottleneck, R*C)
            ])

        self.nets = [
            self.embed,
            self.policy_head,
            self.value_head,
        ]

        self.initialize()

    def initialize(self):
        for net in self.nets:
            for _, params in net.state_dict().items():
                try:
                    nn.init.xavier_uniform_(params)
                except:
                    nn.init.uniform_(params, 0.1)

    def forward(self, state):
        embedding = self.embed(state)

        value = self.value_head(embedding)
        logits = self.policy_head(embedding)

        return logits, value