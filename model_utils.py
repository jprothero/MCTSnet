import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import config
import numpy as np
from IPython.core.debugger import set_trace
from MCTSnet_model import MCTSnet

def setup_optim(model):
    model_opt = optim.SGD(model.parameters(),
        lr=config.LR,
        momentum=config.MOMENTUM)

    return model_opt

def save_model(model, model_name="MCTSnet"):
    torch.save(model, "checkpoints/models/%s.t7" % model_name)
    print('Successfully saved parameters for the best %s' % model_name)

def load_model(model_name="MCTSnet"):
    try:
        model = torch.load('checkpoints/models/%s.t7' % (model_name))
        print('Successfully loaded parameters for the best %s' % model_name)
        return model
    except:
        model = MCTSnet(config.R, config.C)
        print('Initializing new Network Weights for %s net' % model_name)

    return model
