from MCTSnet import MCTSnet
from Connect4 import Connect4
import config
import numpy as np
import pickle
from IPython.core.debugger import set_trace
import torch
import utils
from MCTSnet_trainer import Trainer

import model_utils

def lr_find():
    net = model_utils.load_model()

    memories = utils.load_memories()

    trainer = Trainer()

    return trainer.lr_find(net, memories)