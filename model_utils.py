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

def save_model(model):
    torch.save(model, "checkpoints/models/MCTSnet.t7")
    print('New best model saved')

def load_model(cuda=torch.cuda.is_available()):
    try:
        model = torch.load('checkpoints/models/MCTSnet.t7')
        print('Loaded best model')
        return model
    except Exception as e:
        print(e)
        model = MCTSnet(config.R, config.C)
        print('Initializing new weights')

    if cuda:
        model = model.cuda()

    return model
