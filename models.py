import torch
from torch import nn
from wrn import WideResNet
import model_config
import config

class EmbNet(nn.Module):
    def __init__(self):
        super(EmbNet, self).__init__()

        ng = model_config.NUM_GROUPS
        k = model_config.MULTIPLIER
        n = model_config.NUM_LAYERS

        self.emb_net = WideResNet(num_groups=ng, N=n, k=k, in_channels=config.CH)
        
    def forward(self, state):
        return self.emb_net(state)

class PolicyNet(nn.Module):
    def __init__(self, policy_bottleneck=2):
        super(PolicyNet, self).__init__()

        ng = model_config.NUM_GROUPS
        k = model_config.MULTIPLIER
        n = model_config.NUM_LAYERS

        R, C = config.R, config.C

        num_channels = model_config.BASE_CHANNELS*ng*k

        self.policy_net = nn.Sequential(*[
            WideResNet(num_groups=ng, N=n, k=k, num_classes=policy_bottleneck,
             in_channels=num_channels),
            nn.Linear(policy_bottleneck, R*C)
        ])

        self.value_head = ValueHead()

    def forward(self, emb):
        return self.policy_net(emb), self.value_head(emb)

class BackupNet(nn.Module):
    def __init__(self):
        super(BackupNet, self).__init__()

        ng = model_config.NUM_GROUPS
        k = model_config.MULTIPLIER
        n = model_config.NUM_LAYERS

        #*2 because it's two embedding concatted + 2 values
        num_channels = model_config.BASE_CHANNELS*ng*k*2 + 2

        self.backup_net = WideResNet(num_groups=ng, N=n, k=k, in_channels=num_channels)
        
    def forward(self, emb):
        return self.backup_net(emb)

class ValueHead(nn.Module):
    def __init__(self, value_bottleneck=2):
        super(ValueHead, self).__init__()

        ng = model_config.NUM_GROUPS
        k = model_config.MULTIPLIER
        n = model_config.NUM_LAYERS

        num_channels = model_config.BASE_CHANNELS*ng*k

        self.value_head = nn.Sequential(*[
            WideResNet(num_groups=ng, N=n, k=k, num_classes=value_bottleneck,
             in_channels=num_channels),
            nn.Linear(value_bottleneck, 1),
            nn.Tanh()
        ])
        
    def forward(self, emb):
        return self.value_head(emb)

