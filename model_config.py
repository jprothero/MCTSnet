#this is always the same
#this refers to the base channels used in the wide residual networks
#changing this doesnt do anything, it's only used to calculate the number of channels
BASE_CHANNELS=16

#these are hyperparameters for the wide residual networks
#increasing them with make the network more expressive at the cost of higher parameters
NUM_LAYERS=1
MULTIPLIER=3
NUM_GROUPS=1

#these are for the policy and value which should be more bottlenecked so that
#they generalize better, these are both the minimum size for efficiency at this point
SMALL_N = 2
SMALL_K = 2
SMALL_NG = 1

POLICY_BOTTLENECK=5
VALUE_BOTTLENECK=5
CONTINUE_BOTTLENECK=5

MAX_SIMS=5