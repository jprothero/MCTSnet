from newMCTS import MCTSnet
from Connect4 import Connect4
import config
import numpy as np
import pickle
from IPython.core.debugger import set_trace

connect4 = Connect4()
actions = connect4.actions
calculate_reward = connect4.calculate_reward
get_legal_actions = connect4.get_legal_actions
transition = connect4.transition
test_valid = connect4.test_valid

root_state = np.zeros(shape=(3, 6, 7), dtype="float32")
iteration = 0

# mcts_net = MCTSnet(actions=actions, calculate_reward=calculate_reward,
#             get_legal_actions=get_legal_actions, transition=transition)

#I suspect that the memories are not correctly propagating the gradient to 
#the networks. Need to inspect that and see if they are working correctly

try:
    memories = pickle.load(
        open("checkpoints/memories.p", "rb"))
except FileNotFoundError:
    print("Memories not found, making new memories.")
    memories = []

print("Number of memories: "+str(len(memories)))
mcts_net = MCTSnet(actions=actions, calculate_reward=calculate_reward,
    get_legal_actions=get_legal_actions, transition=transition, test_valid=test_valid)
# mcts_net.save_training_model()

while True:
    if len(memories) > config.MIN_MEMORIES:
        mcts_net.train_memories(memories)
    else:
        print("Not enough memories ({}), need {}".format(len(memories), config.MIN_MEMORIES))

    memories = mcts_net.self_play(root_state, memories)
    # mcts_net.save_memories()
    # print("Number of memories: "+str(len(memories)))
    # mcts_net.load_training_model()
    # mcts_net = MCTSnet(actions=actions, calculate_reward=calculate_reward,
    #     get_legal_actions=get_legal_actions, transition=transition, trainer=True, 
    #     memories=memories)

    print("Saving memories")
    pickle.dump(memories,
        open("checkpoints/memories.p", "wb"))

    iteration += 1
    print("Iteration Number "+str(iteration))
