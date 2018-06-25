from MCTSnet import MCTSnet
from Connect4 import Connect4
import config
import numpy as np
from IPython.core.debugger import set_trace
import torch
import utils
from MCTSnet_trainer import Trainer
import multiprocessing as mp

import argparse

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

    num_worker = 4
    pool = ctx.Pool(processes=num_worker)

    # multicore trainer
    def multi_trainer():
        res = pool.map(F, [root_state]*(num_worker-1))
        for tup in res:
            # tup[0] === new_memories
            memories.extend(tup[0])
        utils.save_memories(memories)
        trainer.fastai_train(mctsnet.new, memories)
        pool.apply_async(G, (root_state,))

    while iteration <= 500:
        """
        # single process trainer & evaluator
        new_memories, _ = F(root_state)
        memories.extend(new_memories)
        utils.save_memories(memories)
        trainer.fastai_train(mctsnet.new, memories)
        G(root_state)
        """
        multi_trainer()
        iteration += 1
        print("Iteration Number "+str(iteration))

    pool.join()
    pool.close()
