import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models import create_net
import config
import numpy as np
from IPython.core.debugger import set_trace

def load_models():
    models = dict()
    models["emb"] = load_model("emb")
    models["policy"] = load_model("policy")
    models["backup"] = load_model("backup")
    models["readout"] = load_model("readout")

    return models

def setup_optims(models, cuda):
    optims = dict()
    for k, model in models.items():
        optims[k] = optim.SGD(model.parameters(),
                                lr=config.LR,
                                momentum=config.MOMENTUM)

    return optims

def load_model(model_name):
    try:
        model = torch.load('checkpoints/models/%s.t7' % (model_name))
        print('Successfully loaded parameters for the best %s net' %model_name)
        return model
    except:
        model = create_net(model_name)
        print('Initializing new Network Weights for %s net' % model_name)

    return model