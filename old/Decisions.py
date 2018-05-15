import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import torch
import numpy as np
from torch.autograd import Variable
from ipdb import set_trace
from .AlphaZero import AlphaZero
from random import sample

from fastai.imports import *
from fastai.transforms import *
from fastai.learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

from copy import deepcopy as dc

from concurrent.futures import ProcessPoolExecutor as PPE
from concurrent.futures import ThreadPoolExecutor as TPE
from torch.multiprocessing import Pool, set_start_method, get_context
from torch.nn.utils import weight_norm as wn

from torch import optim
#so lets see.. basically what I want controller to be is a nn.Module that has different parts
#so like it has an embed method, policy method, backup, etc
#the issue is that right now it's very coupled to an LSTM
#but it should be simple enough just making it different I guess

#I'm just going to build around max depth, it has some uses, if it doesnt apply I can just make it 1e9 or some large number that 
#cant be reached

#well this isnt as simple as I hoped but what can I expect
#basically what I want to do is call forward with different string directions
#such as controller(x, hidden, phase="embed"), redirects to the embed forward

class Decisions(nn.Module):
    def __init__(self, controller, decisions, decision_order, decision_conditions, mask_conditions, 
        stop_conditions, softmaxs, embeddings=None, num_decisions=None, starting_indices=None, cuda=torch.cuda.is_available()):
        super(Decisions, self).__init__()

        self.has_cuda = cuda

        self.controller = controller

        self.decisions = decisions
        self.decision_order = decision_order

        self.decision_conditions = decision_conditions
        self.mask_conditions = mask_conditions
        self.stop_conditions = stop_conditions

        if num_decisions is None: self.num_decisions = len(decisions) 

        self.embeddings = embeddings
        self.softmaxs = softmaxs

        if starting_indices is None: starting_indices = [0]

    def check_condition(self, az, layer_idx, decision_name):
        condition = self.decision_conditions[decision_name]
        return condition(layer_idx)

    def get_values(self, alpha_zeros):
        cont_outs = []
        for az in alpha_zeros:
            cont_outs.append(az.cont_out)

        cont_outs = torch.cat(cont_outs)

        values = self.value_head(cont_outs.squeeze())

        for az, value in zip(alpha_zeros, values):
            az.value = value.detach().item()

        return az

    def backup(self, az):
        az.backup(az.value)
        return az

    def expand(self, az):
        probas = az.probas
        hidden = az.hidden

        depth = az.curr_node["d"]
        decision_idx = depth % len(self.decision_order)
        decision_name = self.decision_order[decision_idx]
        layer_idx = depth // len(self.decision_order)

        if self.mask_conditions[decision_name] is not None:
            probas = self.mask_conditions[decision_name](layer_idx, probas)

        if az.do_expand:
            az.expand(probas, hidden)

        return az

    def evaluate(self, alpha_zeros, states):
        decision_indices = []

        decision_indices_lists = [[] for _ in range(len(self.decision_order))]

        for i, az in enumerate(alpha_zeros):
            decision_indices_lists[az.decision_idx].append(i)
        
        cont_outs = self.controller(states)
        cont_outs = cont_outs.squeeze(0)

        for i, decision_indices in enumerate(decision_indices_lists):
            if len(decision_indices) > 0: 
                specific_cont_outs = cont_outs[decision_indices]
                logits = self.softmaxs[i](specific_cont_outs).view(len(specific_cont_outs), -1)
                probas = F.softmax(logits, dim=1)
                azs = [alpha_zeros[i] for i in decision_indices]
                for az, p in zip(azs, probas):
                    az.probas = p.squeeze().detach().data
                    if self.has_cuda:
                        az.probas = az.probas.cpu()
                    
                    az.probas = az.probas.numpy()

        for az, cont_out in zip(alpha_zeros, cont_outs):
            az.cont_out = cont_out.unsqueeze(0)
            if az.curr_node["d"] < az.max_depth:
                az.do_expand = True
            else:
                az.do_expand = False
        
    def simulate(self, az):
        trajectory = az.select(self.starting_indices, self.decision_order)

        depth = az.curr_node["d"]

        layer_idx = depth // len(self.decision_order)
        decision_idx = depth % len(self.decision_order)
        decision_name = self.decision_order[decision_idx]

        while True:
            skip_curr = self.check_condition(az, layer_idx, decision_name)
            if not skip_curr:
                break
            else:
                az.curr_node["d"] += 1
                depth = az.curr_node["d"]
                layer_idx = depth // len(self.decision_order)
                decision_idx = depth % len(self.decision_order)
                decision_name = self.decision_order[decision_idx]

        az.trajectory = trajectory
        az.decision_idx = decision_idx

        return az

    def get_memories(self, az): return az.new_memories

    def move_choice(self, az):
        d = az.curr_node["d"]
        
        decision_idx = d % len(self.decision_order)
        starting_idx = self.starting_indices[decision_idx]
        name = self.decision_order[decision_idx]
        choice_idx, visits = az.select_real() 

        emb_idx = starting_idx + choice_idx
        az.decisions[name].append(choice_idx)

        az.new_memories.append({
            "search_probas": torch.from_numpy(visits).float()
            , "trajectory": dc(az.real_trajectory)
            , "decision_idx": decision_idx
        })

        az.real_trajectory.append(emb_idx)
        az.trajectory = az.real_trajectory

        if d < az.max_depth-1:
            az.done = False
        else:
            az.done = True

        return az

    def make_architecture_mp(self, kwargs):
        num_archs, num_sims = kwargs["num_archs"], kwargs["num_sims"]
        
        alpha_zeros = [AlphaZero(max_depth=self.num_layers*len(self.decision_order)) for _ in range(num_archs)]

        decisions = dict()
        for name in self.decision_order:
            decisions[name] = []

        for az in alpha_zeros:
            az.real_trajectory = []
            az.decisions = dc(decisions)
            az.new_memories = []

        del decisions

        final_alpha_zeros = []

        i = 0
        while True:
            print(f"Choice {i} of {self.num_decisions-1}")
            for _ in range(num_sims):
                alpha_zeros = list(map(self.simulate, alpha_zeros))

                self.evaluate(alpha_zeros)

                alpha_zeros = list(map(self.expand, alpha_zeros))

                self.get_values(alpha_zeros)

                alpha_zeros = list(map(self.backup, alpha_zeros))

            not_done_alpha_zeros = []
            for az in alpha_zeros:
                az = self.move_choice(az)
                if az.done:
                    final_alpha_zeros.append(az)
                else:
                    not_done_alpha_zeros.append(az)

            alpha_zeros = not_done_alpha_zeros

            i += 1
            
            if len(alpha_zeros) == 0:
                break

        new_memories = list(map(self.get_memories, final_alpha_zeros))

        return new_memories

    

    