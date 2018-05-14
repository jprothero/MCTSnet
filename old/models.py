import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import config

import model_configs

from IPython.core.debugger import set_trace

from wide_resnet import Wide_ResNet

class Controller(nn.Module):
    def __init__(self, num_options):
        super(Controller, self).__init__()

        self.embed_net = Wide_ResNet(7, 2, .3)
        self.simulate_net = Wide_ResNet(7, 2, .3, num_options)
        self.backup_net = Wide_ResNet(7, 2, .3)
        self.readout_net = Wide_ResNet(7, 2, .3, num_options)

    def forward(self, x, hidden=None, phase=None):
        #so lets see.. we for example are going to get in a connect four state
        #(x=state, hidden=None, phase=embed)
        #so yeah lets say we have a connect4 state (batch_size, 3, 6, 7)
        #we want to return (batch_size, 3, 6, 7)
        #what about hidden? will embed use hidden?
        #I dont think so, unless we want to embed different information depending on the "history"
        #which we may in theory I guess

        #we need to think about what we want 
        #so my idea for an MCTSnet was to basically make it an unrolled forget gate only LSTM,
        #except with convs since it can use them.

        #so lets see, let me go over the main concept
        #the concept is that we have an embedding/processing that looks at a state
        #and processes it into a useful representation we want
        #if we have hidden I guess we can use past information to choose how we embed
        #but how would we even do that?
        #the LSTM method is basically do two layers, add them together, then do an activation
        #that implies 50/50 weighting between the two layers, and it is only an addition combine
        #in theory maybe we could do like a concat, then have a third layer do the combine
        #then it can in theory learn to use whatever is useful
        #that sounds good, and I've seen similar things, condensing the number of layers using 
        #conv1x1's

        #so lets assume we do use hidden for everything
        #but the issue is what do we quantify as hidden

        #basically the hidden is the memory from the current trajectory

        #well basically what we need to know is 

        #well what types of "memory" are we going to have

        #right now we make an "embedding"
        #then we choose a policy with that embedding, and run simulations
        #then we backup from a root embedding to all of the parents,
        #i.e. we can do a kind of inverse LSTM hidden states, where we do a forget gate
        #from the last embedding to the first
        #and that in theory updates the embedding so that we better know 
        #what to simulate next time

        #now, it probably would also be ideal to have a working memory which keeps track of where we're
        #going for the simulation, but the issue is that in theory everything is handled from the backups
        #and also, I think maybe decoupling the representation from the hidden instructions might be good

        #because if you think about it, we embed, then later we use the additional simulation information in additions
        #for simplicity we could have it update the original embedding though

        #so we are given an input state, we embed it to whatever number of channels we want,
        #then we use those embeddings to choose what simulation policy to do
        #does the simulate need any memory or can it be handled in the embedding?
        #the simulate is basically used the same time as the embed
        #so basically what would happen is we reach a leaf node, we embed/create a representation,
        #then we send those off to the value and policy heads,
        #and continue with alpha zero like normal, then we update all previous embeddings
        #based on the value and subsequent embeddings
        #hmmm 
        #well the issue is that we need to update the policy every time we reach it

        #and again, we're really creating our own variation, so whatever we think makes sense we can do

        #I like the idea of doing one embedding, then updating it, or simulataneously updating a history
        #I think decoupling the state embedding and the history might be good, but then it slightly complicates "reading it out"
        #basically we're encoding some instructions for what the simulate should do based on the embedding

        #so lets think about the flow for a bit:
        #we take in a state, we're going through the MCTS algo
        #we hit a leaf node, i.e. a state we haven't seen before from this trajectory
        #then we need to "embed" it, i.e. change the number of channels and change the representation
        #into a more useful one. It should probably be at least the same size as the input, so that we can
        #in theory fit all of the information in (especially if it's spatial)
        
        #then what. If we're using alpha zero UCT we generate a value and a policy with the embedding, and we backup
        #basically what MCTSnet suggests is that we backup the embeddings, and value, and use that to update previous embeddings
        #what that would mean that next time around, i.e. when we're doing UCT select, we want to update 
        #the policy prior and the value, or perhaps just the policy

        #so basically whenever we do a backup, we also run a policy on the updated embedding, and we update
        #the policy. the issue with that is that it's a lot of overhead for just changing a policy prior

        #I mean what I'd like to maybe do is have the MCTSnet and then just train it with alpha zero
        #so we keep more of a flow from the real paper,
        #i.e. we embed, uct_select, backup, and readout at the end
        #and then basically the alphazero is blackbox on top
        #so how would we do that.
        #basically we would treat the MCTSnet as decisions 
        #and we get to the simulate decisions,
        #and we try different options, given a certain prior,
        #and we have them match...

        #hm

        #so I mean basically what we want to do is 

        #So yeah I mean we're kind of at a crossroads
        #we can either try to make a net optimized for alpha zero,
        #or we can try to make something with more flexibility and a bit different, but idk if 
        #it will train with alpha zero on top

        #well lets see, what decisions would we make with the MCTSnet alpha zero
        #basically we would select what to UCT select
        #so at every simulate step we would produce a UCT distribution, 
        #and we would 

        #so basically whenever we simulate we create a distribution based on the current embedding, and 
        #the history,

        #and we could have many different histories, lets think about what would be useful
        #there's the current simulate memory, so based on the embeddings and histories we've seen so far,
        #what is the best way to proceed, so that would kind of be the "forward" memory

        #then there is the backup history creator which basically takes the last embedding and the result, and 
        #updates a history associated with those embeddings

        #so we have a growing node memory that updates what the result of the simulations were

        #we could also have different memory contexts, such as game wide, i.e. we have a memory that is updated also,
        #and probably we start our simulation memory from that working start.

        #all of these make it more complicated though of course

        #and before I do any of this I want to see if I can do this with the alpha zero algo

        #basically for the UCT select phase, we generate a policy, but instead of generating it we run simulations using alpha zero
        #so we're kind of doing a meta UCT
        #we have a choice, and we do a prior and a value (based on the embedding/history)
        #then we do a normal alpha zero algo, and we for example proceed to the next sim, and do the same,
        #then update that.

        #so lets see, how would we learn an arbitrary function to use memory and an x
        #basically we can do a concat, thena conv1x1, then it will look for whatever patterns it wants, 
        #and probably it will encourage the net to have them align vertically, which might make things easier

        #so anyways we get to a simulation, we need to make a policy, and we want to find it from alpha zero,
        #so we do an alpha zero move select

        #the issue: it would be a bandit context, because we dont know what the future simulations would be 
        #although actually, I guess we are simulating internally right
        #since it's a two player game we simulate one move, then the other, and we backup those
        #so we can keep doing simulations until a condition is met.

        #question: do we actually need to do the complicated conv combining?
        #in theory I guess we could do what LSTM's do and keep them separate and in theory they learn their magnitudes
        #and implicit combine ratios. basically it would be that each part "votes" and they are implicitly weighed
        #by the train algorithm. in theory maybe we could make it easier for the algo if we show it everything at once 
        #and let it explicitly combine them, but it would make things a bit more complicated.

        #so what the heck am I trying to do.

        #basically the question is do we want to do the MCTSnet thing or not
        #this random dude shouldnt influence my decision, I should do what I think is better
        #I would like to discover a more effective alpha zero system, that was the root of
        #my initial interest in MCTSnets

        #basically we can either make a slightly better alpha zero architecture (pretty intellectually vapid, idk what else to change)
        #or we can give more freedom and utilize memory more extensively

        #so maybe the MCTSnet is the only way to go
        #idk, I just really dont know what else to change with the original, so maybe this would be a good potential alternative
        #it basically would be we allow the system to learn to create a memory telling it what to simulate,
        #and we use alpha zero vanilla to fine tune.

        #what are some potential alternatives?
        #we can try to find a better architecture for the alpha zero algo, 
        #such as maybe using more attention or an external memory
        
        #and how about my idea of musical programs using recursion
        #basically the idea is you give the system the tools of recursion, and that should
        #give simplicity and power.

        #for that it would basically be we have a pointer network type thing, where we dynamically generate things
        #and we have a controller that controls the flow of the network

        #could that apply to alpha zero?
        #its basically like we simplify the information storage and flow by calling modules recursively
        #rather than new ones

        #so for example, what if we had a random function, that takes in an embedding
        #and does some random modification of interest, i.e. it transitions it to an embedding of an interesting state,
        #then we for example send that off to the value network, and do a backup of that modified embedding and the original embedding 
        #to a "simulation history", then we can use that simulation history to choose a final move

        #so basically what it would be doing is have everything work in the "representation" space
        #we dont explicitly sample policies and use them, we just imagine representations, and use that 
        #to backup the original one. the imagined representations would need to be the same domain as the real ones though
        
        #but lets see. at the end of the day if we want to generate a policy, and make it fully differentiable, what would we do
        
        #maybe we can train to map embeddings to a policy, then we produce a weighted mix (attention) of the embeddings,
        #and use that as the real embedding. 

        #so the idea is basically to take MCTSnet and make it fully differentiable

        #so we have some input state S
        #S -> emb_net -> embedding
        #then we do "simulations" on it
        #i.e. we do a simulation_net: embedding -> simulation_net -> imagined_embedding
        #then we predict the value for the imagined embedding, and the embedding, and we use (emb1, emb2, val1, val2) 
        # -> backup_net -> update_history
        #a question is, do we want to imagine from imaginings? I think not, since in theory we can think about any future embedding
        #the question is how many ponders do we want to do, we can make it a threshhold, i.e. we predict how long we want to do it
        #how can we train that though. 
        #ideally we want to make it fully differentiable.
        #but for that we would basically need to both "simulate and not simulate"
        #so idk, we could for example mix based on the values or something?
        #or we could do a softmax based on the history and the state, and learn the percentage

        #so we also dictate what the simulation_net does based on the simulation history


        #so lets think about this some more

        #state S -> emb_net -> emb E -> value_net, simulation_net -> value_net| (emb1, emb2, v1, v2) -> backup_net
        #what uses the "simulation history / memory", in theory all of them right? basically we're creating an embedding which tells
        #the system instructions, having the memory at any given point seems like it could be helpful

        if phase is "embed":
            x, hidden = self.embed(x, hidden)
        elif phase is "simulate":
            x, hidden = self.simulate(x, hidden)
        elif phase is "backup":
            x, hidden = self.backup(x, hidden)
        elif phase is "readout":
            x, hidden = self.readout(x, hidden)
        else:
            raise Exception("Unknown phase")

        return x, hidden

    def embed(self, x, hidden):
        pass

    def simulate(self, x, hidden):
        pass

    def backup(self, x, hidden):
        pass

    def readout(self, x, hidden):
        pass

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