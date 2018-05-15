import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from copy import copy

from IPython.core.debugger import set_trace
import config

import random
from random import shuffle

import hashlib

import pickle
import sys

import utils
import model_utils

from AlphaZero import AlphaZero

np.seterr(all="raise")

class MCTSnet:
    def __init__(self,
                 actions,
                 get_legal_actions,
                 transition_and_evaluate,
                 best=False):
        utils.create_folders()
        self.has_cuda = config.CUDA

        self.actions = actions
        self.get_legal_actions = get_legal_actions
        self.transition_and_evaluate = transition_and_evaluate

        self.new = model_utils.load_model()
        self.best = model_utils.load_model()
        
        if self.has_cuda: 
            self.new = self.new.cuda()
            self.best = self.best.cuda()

    def self_play(self, root_state, best_only=True, 
        num_episodes=config.NUM_EPISODES, deterministic=False, vs_human=False):
        self.best.eval()
        self.new.eval()

        best_az = AlphaZero()
        new_az = AlphaZero()

        starting_player = 1

        if best_only:
            order = [self.best, self.best]
            name_order = ["best", "best"]
            az_order = [best_az, new_az]
        elif vs_human:
            order = [None, self.best]
            name_order = ["human", "best"]
            az_order = [None, best_az]
        else:
            if np.random.uniform() > .5:
                order = [self.best, self.new]
                name_order = ["best", "new"]
                az_order = [best_az, new_az]
            else:
                order = [self.new, self.best]
                name_order = ["new", "best"]
                az_order = [new_az, best_az]

        scoreboard = {
            "new": 0
            , "best": 0
            , "draws": 0
        }

        memories = []
        np.set_printoptions(precision=3)
        # sim_state_np[0] + sim_state_np[1] + np.reshape(policy, (6, 7)) #for debugging
        for _ in tqdm(range(num_episodes)):
            game_over = False

            state_np = np.array(root_state)
            state = self.convert_to_torch(root_state).unsqueeze(0)

            best_az.reset()
            new_az.reset()

            if deterministic:
                best_az.T = 0
                new_az.T = 0

            starting_player = (starting_player+1)%2
            curr_player = starting_player
            i = 0
            episode_memories = []
            while not game_over:
                if i > 0: curr_player = (curr_player+1)%2

                net = order[curr_player]
                az = az_order[curr_player]
                other_az = az_order[(curr_player+1)%2]
                # if ((state_np[0] + state_np[1]) > 1).any():
                
                if vs_human and name_order[curr_player] is "human":
                    action = self.choose_column()
                else:
                    for _ in range(config.NUM_SIMS):
                        #need to change result so that it is updated based on if the player that starting the sim (root state)
                        #matchs
                        sim_state = state.clone() 
                        sim_state_np = np.array(state_np)
                        # print(sim_state_np)
                        # set_trace()

                        sim_state_np, result, sim_over = az.select(sim_state_np, self.transition_and_evaluate)

                        if sim_over and sim_state_np[2][0][0] != starting_player and result != 0:
                            result *= -1

                        sim_state = self.convert_to_torch(sim_state_np).unsqueeze(0)

                        policy, value = net(sim_state)
                        policy = policy.squeeze().detach()
                        if self.has_cuda:
                            policy = policy.cpu()
                        
                        policy = policy.numpy()
                        value = value.detach().squeeze().item()

                        if result is not None:
                            value = result

                        if not sim_over:
                            az.expand(policy, sim_state_np, self.correct_policy)

                        az.backup(value)

                    action, search_probas = az.select_real()

                    if other_az.curr_node["children"] is not None:
                        other_az.curr_node = other_az.curr_node["children"][action]
                        other_az.curr_node["parent"] = None
                    else:
                        other_az.reset()

                    episode_memories.append({
                        "state": torch.tensor(state_np).float(),
                        "search_probas": torch.tensor(search_probas).float(),
                        "curr_player": curr_player
                    })

                # state_np1, result1, game_over1 = self.transition_and_evaluate(state_np, action)
                # if result1 is not None:
                #     set_trace()
                # print(state_np[0])
                # if curr_player == 0:
                #     set_trace()
                state_np, result, game_over = self.transition_and_evaluate(state_np, action)
                assert ((state_np[0] + state_np[1]) < 2).all()
                state = self.convert_to_torch(state_np).unsqueeze(0)
                # print(name_order[curr_player], "Best Only: "+str(best_only))

                i += 1

                #cool visualization of the board, good for debugging
                # print(state_np[0]+state_np[1])

            # print("Result {}\nCurr Player: {}".format(result, curr_player))
            # set_trace()

            # set_trace()
            if result != 0:
                scoreboard[name_order[curr_player]] += 1
            else:
                scoreboard["draws"] += 1                                

            for memory in episode_memories:
                if memory["curr_player"] != curr_player and result != 0:
                    result *= -1
                memory["result"] = result

            memories.extend(episode_memories)

        if not best_only:
            print("{} Wins: {}\n{} Wins: {}\n Draws: {}".format(name_order[0], scoreboard[name_order[0]],
                name_order[1], scoreboard[name_order[1]], scoreboard["draws"]))

        return memories, scoreboard

    def correct_policy(self, policy, state):
        # state = np.reshape(state, newshape=tuple(state.shape[1:]))
        mask = np.zeros(shape=policy.shape)
        legal_actions = self.get_legal_actions(state[:2])
        mask[legal_actions] = 1
        policy = policy * mask

        pol_sum = np.sum(policy) * 1.0

        if pol_sum != 0:
            policy = policy / pol_sum

        return policy

    def convert_to_torch(self, state):
        state = torch.tensor(state)
        if self.has_cuda:
            state = state.cuda()

        return state

    def tournament(self, root_state):
        _, scoreboard = self.self_play(root_state, best_only=False, 
            num_episodes=config.NUM_TOURNAMENT_EPISODES, deterministic=True)

        if scoreboard["new"] > scoreboard["best"]*config.SCORING_THRESHOLD:
            model_utils.save_model(self.new)
            self.best = model_utils.load_model()
        elif scoreboard["new"]*config.SCORING_THRESHOLD < scoreboard["best"]:
            self.new = model_utils.load_model()

    def choose_column(self):
        while True:
            try:
                inp = int(input("Pick a column, 1-7: "))
                inp -= 1
                return inp
            except Exception:
                print("Invalid choice.")