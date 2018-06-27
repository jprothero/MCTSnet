from MCTSnet import MCTSnet
from Connect4 import Connect4
import config
import numpy as np
from IPython.core.debugger import set_trace
import torch
import utils
from MCTSnet_trainer import Trainer
import torch.multiprocessing as mp
from torch.multiprocessing import set_start_method

import argparse

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument('--func', default='self_play', help='Choose between self_play and play_minmax')
args = parser.parse_args()

if __name__ == '__main__':
    ctx = mp.get_context('spawn')

    connect4 = Connect4()
    actions = connect4.actions
    get_legal_actions = connect4.get_legal_actions
    transition_and_evaluate = connect4.transition_and_evaluate

    root_state = np.zeros(shape=(3, 6, 7), dtype="float32")
    iteration = 0

    mctsnet = MCTSnet(actions=actions, get_legal_actions=get_legal_actions,
        transition_and_evaluate=transition_and_evaluate)

    memories = utils.load_memories()
    trainer = Trainer()
    # define function references
    funcs = {'self_play': mctsnet.self_play, 'play_minmax': mctsnet.play_minmax}
    F = funcs[args.func]
    G = mctsnet.tournament

    num_worker = 2

    while iteration <= 500:
        """
        # single process trainer & evaluator
        new_memories, _ = F(root_state)
        memories.extend(new_memories)
        utils.save_memories(memories)
        trainer.fastai_train(mctsnet.new, memories)
        G(root_state)
        """
        Queue = ctx.Queue()
        Event = ctx.Event()
        workers = []
        for i in range(num_worker):
            worker = ctx.Process(target=F, args=(root_state,Queue,Event))
            workers.append(worker)

        for worker in workers:
            worker.start()

        ended_worker = 0
        while ended_worker < num_worker:
            try:
                ret = Queue.get()
            except RuntimeError:
                print('RuntimeError: received 0 items of ancdata')
                break
            if ret is None:
                ended_worker += 1
            else:
                memories.extend(ret)

        Event.set()
        for worker in workers:
            worker.join()

        utils.save_memories(memories)
        trainer.fastai_train(mctsnet.new, memories)

        # tournament between new model and best best model
        G(root_state)

        iteration += 1
        print("Iteration Number "+str(iteration))
