from MCTSnet import MCTSnet
from Connect4 import Connect4
import config
import numpy as np
import cProfile

connect4 = Connect4()
actions = connect4.actions
calculate_reward = connect4.calculate_reward
get_legal_actions = connect4.get_legal_actions
transition = connect4.transition

root_state = np.zeros(shape=(6, 7))
iteration = 0

def one_iteration():
    mcts_net = None
    if (iteration % config.EPISODES_BEFORE_MODEL_RESET) == 0:
        if mcts_net is not None:
            mcts_net.train_memories()
        mcts_net = MCTSnet(actions=actions, calculate_reward=calculate_reward,
                           get_legal_actions=get_legal_actions, transition=transition)
    mcts_net.self_play(root_state=root_state)

if __name__ == "__main__":
    cProfile.run("one_iteration()")
