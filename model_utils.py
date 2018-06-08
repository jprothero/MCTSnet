import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import config
import numpy as np
from IPython.core.debugger import set_trace
from models import EmbNet, PolicyNet, BackupNet

def setup_optim(nets):
    params = []
    for _, net in nets.items():
        params += list(net.parameters())

    model_opt = optim.Adam(params,
        lr=config.LR,
        weight_decay=1e-6,
        amsgrad=True)

    return model_opt

# def save_model(model):
#     torch.save(model, "checkpoints/models/MCTSnet.t7")
#     print('New best model saved')

# def load_model(cuda=torch.cuda.is_available()):
#     try:
#         model = torch.load('checkpoints/models/MCTSnet.t7')
#         print('Loaded best model')
#         return model
#     except:
#         model = MCTSnet(config.R, config.C)
#         print('Initializing new weights')

#     if cuda:
#         model = model.cuda()

#     return model

def save_model(nets):
    for name, net in nets.items():
        torch.save(net, "checkpoints/models/{}_net.t7".format(name))

    print('New best model saved')

def load_model(cuda=torch.cuda.is_available()):
    nets = dict()
    try:
        nets["emb"] = torch.load('checkpoints/models/emb_net.t7')
        nets["policy"] = torch.load('checkpoints/models/policy_net.t7')
        nets["backup"] = torch.load('checkpoints/models/backup_net.t7')

        print('Loaded best model')
        return model
    except:
        nets["emb"] = EmbNet()
        nets["policy"] = PolicyNet()
        nets["backup"] = BackupNet()

        print('Initializing new weights')

    if cuda:
        for _, net in nets.items():
            net = net.cuda()

    return nets
