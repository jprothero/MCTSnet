from MCTSnet import MCTSnet
from Connect4 import Connect4
import config
import numpy as np

connect4 = Connect4()
actions = connect4.actions
calculate_reward = connect4.calculate_reward
get_legal_actions = connect4.get_legal_actions
transition = connect4.transition

root_state = np.zeros(shape=(6, 7))

mcts_net = MCTSnet(actions=actions, calculate_reward=calculate_reward,
            get_legal_actions=get_legal_actions, transition=transition)

mcts_net.play_cpu(root_state)