import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from models import create_net
import config
import numpy as np
from IPython.core.debugger import set_trace

def setup_mcts(io, do_load, cuda, trainer=True):
    if trainer:
        name = "training"
    else:
        name = "best"
    if do_load:
        mcts = load_model("mcts_{}".format(name), io)
    else:
        models["emb"] = None
        models["readout"] = None
        models["policy"] = None
        models["backup"] = None

    for model_name, model in models.items():
        if model is None:
            models[model_name] = create_net(model_name)

    train_mode(models)
    if cuda:
        activate_cuda(models)

    return models

def setup_models(io, do_load, cuda, trainer=True):
    models = dict()
    if trainer:
        name = "training"
    else:
        name = "best"
    if do_load:
        models["emb"] = load_model("emb_{}".format(name), io)
        models["readout"] = load_model("readout_{}".format(name), io)
        models["policy"] = load_model("policy_{}".format(name), io)
        models["backup"] = load_model("backup_{}".format(name), io)
    else:
        models["emb"] = None
        models["readout"] = None
        models["policy"] = None
        models["backup"] = None

    for model_name, model in models.items():
        if model is None:
            models[model_name] = create_net(model_name)

    train_mode(models)
    if cuda:
        activate_cuda(models)

    return models


def setup_optims(models, cuda):
    optims = dict()
    for k, model in models.items():
        if config.OPTIM == "adam":
            optims[k] = optim.Adam(model.parameters(),
                                    lr=config.lrs[k],
                                    weight_decay=config.weight_decays[k])
        else:
            optims[k] = optim.SGD(model.parameters(),
                                   lr=config.lrs[k],
                                #    weight_decay=config.weight_decays[k],
                                   momentum=config.MOMENTUM)

    return optims


def activate_cuda(models):
    for k, model in models.items():
        models[k] = model.cuda()


def train_mode(models):
    for _, model in models.items():
        model.train()
        assert model.training


def eval_mode(models):
    for _, model in models.items():
        model.eval()
        assert not model.training


def load_model(model_name, io):
    try:
        model = torch.load('checkpoints/models/%s.t7' % (model_name))
        io.cprint('Loading Parameters from the last trained %s Model' %
                  model_name)
        return model
    except:
        io.cprint('Initialize new Network Weights for %s' % model_name)
        pass
    return None


def cast_to_torch(input, cuda):
    input = torch.from_numpy(np.array(input, dtype="float32"))
    if cuda:
        return cast_to_cuda(input)
    else:
        return cast_to_variable(input)


def cast_to_cuda(input):
    if type(input) == type([]):
        for i in range(len(input)):
            input[i] = cast_to_cuda(input[i])
    else:
        return input.cuda()
    return input


def cast_to_variable(input):
    if type(input) == type([]):
        for i in range(len(input)):
            input[i] = cast_to_variable(input[i])
    else:
        return Variable(input)

    return input
