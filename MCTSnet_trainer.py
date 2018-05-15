from random import sample
import config
import torch
import torch.nn.functional as F

from fastai.imports import *
from fastai.transforms import *
from fastai.learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *

from torch import optim

import numpy as np
from ipdb import set_trace

class Trainer:
    def __init__(self, cuda=torch.cuda.is_available()):
        self.fake_data = create_fake_data(num_batches=config.NUM_BATCHES_PER_TRAINING)
        self.has_cuda = cuda

    def train_wrapper(self, _, __):
        return self.train(self.net, self.memories)

    def train(self, net, memories):
        net.forward = net.real_forward
        net.eval()
        minibatch = sample(memories, min(config.BATCH_SIZE, len(memories)))

        states = []
        results = []
        search_probas = []

        for memory in minibatch:
            states.append(memory["state"])
            results.append(memory["result"])
            search_probas.append(memory["search_probas"])

        states = torch.cat(states, dim=0)
        results = torch.tensor(results).float().unsqueeze(-1)
        search_probas = torch.cat(search_probas)

        if self.has_cuda:
            states = states.cuda()
            results = results.cuda()
            search_probas = search_probas.cuda()
            net = net.cuda()

        policies, values = net(states)
        policies = policies.view(-1)

        value_loss = F.mse_loss(values, results)
        policy_loss = -search_probas.unsqueeze(0) @ torch.log(policies.unsqueeze(-1))
        policy_loss /= len(minibatch)

        total_loss = value_loss + policy_loss

        net.forward = lambda x: x

        return total_loss

    def fastai_train(self, net, memories, num_cycles=10, epochs=1):
        self.memories = memories
        self.net = net

        if len(memories) < config.MIN_MEMORIES:
            print("Have {} memories, need {}".format(len(memories), config.MIN_MEMORIES))
            return

        net_wrapped = FastaiWrapper(model=net, crit=self.train_wrapper)
        learner = Learner(data=self.fake_data, models=net_wrapped)
        learner.crit = net_wrapped.crit
        learner.opt_fn = optim.Adam
        learner.model.train()

        learner.model.real_forward = learner.model.forward

        learner.model.forward = lambda x: x
        learner.fit(5e-3, epochs, wds=1e-7) #was 7e-2

        learner.model.forward = learner.model.real_forward

        del self.memories
        del self.net
        del net_wrapped.model.real_forward
        del net_wrapped

def create_fake_data(num_batches=config.TRAINING_LOOPS):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')

    num_samples = 64*num_batches
    trn_X = np.zeros(shape=(num_samples, config.CH, config.R, config.C))
    trn_y = np.zeros(shape=(num_samples, 1))
    val_X = np.zeros(shape=(num_samples//6, config.CH, config.R, config.C))
    val_y = np.zeros(shape=(num_samples//6, 1))
    trn = [trn_X, trn_y]
    val = [val_X, val_y]
    fake_data = ImageClassifierData.from_arrays("./data", trn=trn, val=val,
                                    classes=classes)
    return fake_data

class FastaiWrapper():
    def __init__(self, model, crit):
        self.model = model
        self.crit = crit
    
    def get_layer_groups(self, precompute=False):
        return self.model


