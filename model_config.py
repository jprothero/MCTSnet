#this is always the same
#this refers to the base channels used in the wide residual networks
#changing this doesnt do anything, it's only used to calculate the number of channels
BASE_CHANNELS=16

#these are hyperparameters for the wide residual networks
#increasing them with make the network more expressive at the cost of higher parameters
NUM_LAYERS=2
MULTIPLIER=2
NUM_GROUPS=2

#these are for the policy and value which should be more bottlenecked so that
#they generalize better, these are both the minimum size for efficiency at this point
SMALL_N = 1
SMALL_K = 1
SMALL_NG = 1