import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config

import model_configs

from IPython.core.debugger import set_trace

def conv_layer(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1)

class PolicyBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims, add_gate=True, gate_only=False):
        super(PolicyBlock, self).__init__()
        self.add_gate = add_gate

        self.conv1 = conv_layer(in_dims, h_dims)
        self.bn1 = nn.BatchNorm2d(h_dims)
        self.relu = nn.ReLU()
        if out_dims is None:
            out_dims = h_dims
        self.conv2 = conv_layer(h_dims, h_dims)
        self.bn2 = nn.BatchNorm2d(h_dims)

        self.policy = nn.Linear(h_dims*6*7, out_dims)

        if self.add_gate:
            self.gate = nn.Linear(in_dims*6*7, out_dims)
            self.tanh = nn.Tanh()
            # self.conv3 = conv_layer(h_dims, out_dims)
            # self.bn3 = nn.BatchNorm2d(out_dims)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x += residual
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        out = F.softmax(self.policy(x.view(config.BATCH_SIZE, -1)), dim=1)
        
        if self.add_gate:
            out *= self.tanh(self.gate(x.view(config.BATCH_SIZE, -1)))

        return out

class MCTS(nn.Module):
    def __init__(self, num_filters=32, channels=3):
        super(MCTS, self).__init__()
        self.block = ResBlock
        
        self.relu = nn.ReLU()
        #the input for the state will be (1, 3, 6, 7)
        #can I reshape the memory vector and just concat it?
        #probably I can do 3

        #memory will be an identical state to the board, and can consider concatting additional
        #filters. one issue with this is that it will assume that the memory will take the
        #same structure as the board. this isn't necesssarily bad though.
        self.relu = nn.ReLU()

        self.unsqueeze = conv_layer(4, num_filters)
        self.bn1 = nn.BatchNorm2d(num_filters)

        #lots of things could be considered, such as number of res blocks for each head,
        #number of the squeeze/unsqueeze convs, number of blocks for the main body, etc
        self.res1 = self._make_layer(num_filters, num_filters, ResBlock)
        self.res2 = self._make_layer(num_filters, num_filters, ResBlock)
        self.res3 = self._make_layer(num_filters, num_filters, ResBlock)
        self.res4 = self._make_layer(num_filters, num_filters, ResBlock)
        self.res5 = self._make_layer(num_filters, num_filters, ResBlock)
        self.res6 = self._make_layer(num_filters, num_filters, ResBlock)
        self.res7 = self._make_layer(num_filters, num_filters, ResBlock)

        # self.img1 = self._make_layer(num_filters, num_filters, ResBlock)
        self.squeeze = conv_layer(num_filters, 3)
        self.bn2 = nn.BatchNorm2d(3)
        self.img1 = self._make_layer(3, 3, ResBlock)
        self.img2 = self._make_layer(3, 3, ResBlock)
        self.img3 = self._make_layer(3, 3, ResBlock)        

        #for _ in range(model_conf.NUM_HEAD_BLOCKS):
        self.valid1 = self._make_layer(3, 3, ResBlock)
        self.valid = self._make_layer(3*6*7, num_filters, ValueHead, out_dims=1, add_gate=True)

        #for _ in range(model_conf.NUM_HEAD_BLOCKS):
        self.value = self._make_layer(32*6*7, num_filters, ValueHead, out_dims=1, 
            add_gate=True)

        #for _ in range(model_conf.NUM_HEAD_BLOCKS):
        self.policy = self._make_layer(32, 32, PolicyBlock, 
            out_dims=42, add_gate=True)

    def _make_layer(self, in_dims, h_dims, block, out_dims=None, add_gate=False, gate_only=False):
        layers = []
        layers.append(block(in_dims, h_dims, out_dims, add_gate, gate_only))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, is_real_list=None):
        #A way I could change this is that I could pass one state, and have it
        #make an imagined output, complete with basically everything now, but 
        #it does the imagined part multiple times and averages the results

        #could add a reconstruction loss to learn ignoring the noise a bit, idk
        #for each state add random noise to it'
        # state_flattened = exploratory_state.data.numpy().flatten()
        # std = np.std(state_flattened, axis=0)
        # mean = np.mean(state_flattened, axis=0)
        # noise = torch.from_numpy(np.random.normal(loc=mean, scale=std, size=(state.shape)))
        # seed = starting_seed
        
        #convert state from 4 filters to 32
        residual = x[:, :3]

        # for _ in range(model_conf.NUM_UNSQUEEZE_BLOCKS):
        x = self.unsqueeze(x)
        x = self.bn1(x)
        # x = self.relu(x)

        #imagined_state_processing
        # for _ in range(model_conf.NUM_RES_BLOCKS):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        # x = self.res6(x)
        # x = self.res7(x)

        #can kind of think of this part, the imagined state generator, as the generator
        # for _ in range(model_conf.NUM_SQUEEZE_BLOCKS):
        squeezed_state = self.squeeze(x)
        squeezed_state = self.bn2(squeezed_state)
        # for _ in range(model_conf.NUM_IMAGINED_BLOCKS)
        imagined_state = self.img1(squeezed_state)
        imagined_state = self.img2(imagined_state)
        imagined_state = self.img3(imagined_state)

        valid_input = residual.clone()
        for i, is_real in enumerate(is_real_list):
            if is_real:
                valid_input[i] = residual[i]
            else:
                valid_input[i] = imagined_state[i]

        #can kind of think of this part, the valid discriminator and as the discriminator
        valid1 = self.valid1(valid_input)  
        valid = self.valid(valid1.view(config.BATCH_SIZE, -1))  

        # x = torch.cat([x, imagined_state], axis=1)

        value = self.value(x.view(config.BATCH_SIZE, -1))

        policy = self.policy(x)

        policy = policy*valid + policy*value

        #idea, I am naturally going to be supplying half valid and half invalid inputs to
        #the net, since the first input is a valid one and the second is an invalid one
        #I can have the same block used for both of those, and have that be the training
        #so for the experience replay, I pass in the state with noise, and the is_valid 
        #for the first one is known and doesnt have to be calculated.
        #now one obvious issue with this is that if we keep seeing the same state over and
        #over again it will overfit. also, it will more likely learn the difference between
        #real and fake states, since one is the true input data and the other is imagined data
        #now that isn't necessarily bad, we could probably train the net so that it somehow
        #generates realistic transitions. i.e. we consider the imagined state a generator
        #and the is valid a discriminator, and since we receive 50/50 we will see whether
        #it learns to create more realistic representations. I would need to research gan
        #losses to do that. 
        #for now what I have where I will manually mix the valid / invalid is fine

        #now instead of doing batch training like I am doing right now, I could also have
        #a head which predicts whether or not to do another simulation
        #the value of that simulation could determine whether or not to continue
        #i.e. the value*chance that we should continue
        
        #the issue with this is I can't think of a good way to determine whether to continue
        # or to not. also, right now the second part of the head only takes one 
        # state, so that wouldn't really work. although I think maybe we could concat
        # it on the batch dimension and run it like that. 
        # 
        # It is an interesting idea worth exploring, since basically the net could actively
        # learn how much it wants to think about a state, but for now what I have works 

        #so lets see, the first part takes a state and noise and creates a valid state
        #with that valid state the second part takes the original input, the imagined state
        #and guess the value of the original state, and a policy for the original state
        #the effect the policy will have on the averaging will depend on it's confidence 
        #with the validity and the value
        #so policies with a high value will get more weight and policies with a low validity
        #confidence will get low weight.
        #maybe I can do an average of the two
        #the only thing I really care about averaging is the policy, the rest I will just
        #do supervised learning on. 

        #REMINDER: we want to balance the number of valid states so that the system doesnt get skewed one way or the other. maybe something like that for value too
        
        
        #so okay, what I need to do is have a gan-like setup where the valid input will
        #randomly either receive a real input or an imagined input, and it needs to 
        #be able to discriminate between them. so in 
        # if np.random.uniform() > .5:
        #     valid_inp = residual


        #so a natural question is if we arent doing parameter sharing we could probably
        #just put every in one big block
        #, the difference is I want...
        #this is okay for now. try it out

        #consider zeroing out the noise

        #zero out the noise
        # x[3:] = 0

        # x += residual

        # x = self.res5(x)
        # x = self.res6(x)
        # x = self.res7(x)

        #a shared weight set of residual blocks
        # x = self.res1(x)
        # x = self.res2(x)
        # x = self.res3(x)
        # x = self.res4(x) 
        # -1, -1 
        # -1, 1
        #So yeah that doesn't totally make sense. 
        #the magnitude of the policy averaging is determined by the valid and value heads
        #so where everything is more confident it will get bigger weights

        return valid.squeeze(), value.squeeze(), policy, imagined_state, is_real_list

class ResBlock(nn.Module):
    def __init__(self, in_dims, h_dims, out_dims=None, add_gate=False, gate_only=False):
        super(ResBlock, self).__init__()
        self.add_gate = add_gate
        self.gate_only = gate_only

        self.conv1 = conv_layer(in_dims, h_dims)
        self.bn1 = nn.BatchNorm2d(h_dims)
        self.relu = nn.ReLU()
        if out_dims is None:
            out_dims = h_dims
        self.conv2 = conv_layer(h_dims, out_dims)
        self.bn2 = nn.BatchNorm2d(out_dims)
        if add_gate:
            self.tanh = nn.Tanh()
            self.conv3 = conv_layer(h_dims, out_dims)
            self.bn3 = nn.BatchNorm2d(out_dims)
    
    def forward(self, x):
        residual = x.squeeze()
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.add_gate:
            g = self.conv3(out)
            g = self.bn3(g)
            g = self.tanh(g)
            if self.gate_only:
                return g
            out = g*out

        out += residual
        out = self.relu(out)

        return out

class ValueHead(nn.Module):
    """This network does a readout... (update this)"""
    
    def __init__(self, in_dims, h_dims, out_dims=1, add_gate=False, gate_only=False):
        super(ValueHead, self).__init__()
        self.add_gate = add_gate

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.conv1 = nn.Conv2d(in_planes, num_filters, kernel_size=1, stride=1)
        self.lin1 = nn.Linear(in_dims, h_dims)
        self.bn1 = nn.BatchNorm1d(h_dims)
        self.lin2 = nn.Linear(h_dims, h_dims)
        if self.add_gate:
            self.tanh = nn.Tanh()
            self.gate = nn.Linear(h_dims, out_dims)
        self.lin3 = nn.Linear(h_dims, out_dims)
    
    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        if self.add_gate:
            gated = self.tanh(self.gate(x))
        x = self.lin3(x)
        x = self.tanh(x)
        
        if self.add_gate:
            x *= gated
        
        return x