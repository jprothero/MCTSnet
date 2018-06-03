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

from MinMax import bestMove, fullState_2_gameState

np.seterr(all="raise")

class MCTSnet:
    def __init__(self,
                 actions,
                 get_legal_actions,
                 transition_and_evaluate,
                 cuda=torch.cuda.is_available(),
                 best=False):
        utils.create_folders()
        self.has_cuda = cuda

        self.actions = actions
        self.get_legal_actions = get_legal_actions
        self.transition_and_evaluate = transition_and_evaluate

        self.new = model_utils.load_model(cuda=cuda)
        self.best = model_utils.load_model(cuda=cuda)

        if self.has_cuda:
            self.new = self.new.cuda()
            self.best = self.best.cuda()

    def self_play(self, root_state, best_only=True,
        num_episodes=config.NUM_EPISODES, deterministic=False):
        self.best.eval()
        self.new.eval()

        best_az = AlphaZero()
        new_az = AlphaZero()

        starting_player = 1

        if best_only:
            order = [self.best, self.best]
            name_order = ["best", "best"]
            az_order = [best_az, new_az]
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
                    "state": state.clone(),
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
                #print(state_np[0]+state_np[1]*-1)

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

    def play_minmax(self, root_state, best_only=True,
        num_episodes=config.NUM_EPISODES, deterministic=False):
        self.best.eval()

        best_az = AlphaZero()

        starting_player = 1
        minmax_player = starting_player

        if np.random.uniform() > .5:
            name_order = ["best", "minmax"]
            minmax_player = 1
        else:
            name_order = ["minmax", "best"]
            minmax_player = 0

        scoreboard = {
            "minmax": 0
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

            if deterministic:
                best_az.T = 0
                new_az.T = 0

            starting_player = (starting_player+1)%2
            curr_player = starting_player
            i = 0
            episode_memories = []
            while not game_over:
                if i > 0: curr_player = (curr_player+1)%2

                if  name_order[curr_player] == "best": # case MCTSnet agent to play
                    net = self.best
                    az = best_az
                    # if ((state_np[0] + state_np[1]) > 1).any():

                    for _ in range(config.NUM_SIMS):
                        #need to change result so that it is updated based on if the player that starting the sim (root state)
                        #matchs
                        sim_state = state.clone()
                        sim_state_np = state_np.copy()
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

                    episode_memories.append({
                        "state": state.clone(),
                        "search_probas": torch.tensor(search_probas).float(),
                        "curr_player": curr_player
                    })

                else: # case Minmax agent to play
                    gameState = fullState_2_gameState(state_np, minmax_player)
                    # bestMove return the column to drop a 'stone'
                    gameState_c = gameState.copy()
                    column = bestMove(gameState_c, player=1, opponent=-1)
                    height = np.count_nonzero(gameState[:, column])
                    # action is a index of an spot on the gameboard
                    action = ((config.R-1)-height)*config.C + column

                    if best_az.curr_node["children"] is not None:
                        best_az.curr_node = best_az.curr_node["children"][action]
                        best_az.curr_node["parent"] = None
                    else:
                        best_az.reset()

                # state_np1, result1, game_over1 = self.transition_and_evaluate(state_np, action)
                # if result1 is not None:
                #     set_trace()
                # print(state_np[0])self_play
                # if curr_player == 0:
                #     set_trace()
                state_np, result, game_over = self.transition_and_evaluate(state_np, action)
                assert ((state_np[0] + state_np[1]) < 2).all()
                state = self.convert_to_torch(state_np).unsqueeze(0)
                # print(name_order[curr_player], "Best Only: "+str(best_only))

                i += 1

                #cool visualization of the board, good for debugging
                #print(state_np[0]+state_np[1]*-1)

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

    # def choose_row(self):
    #     while True:
    #         try:
    #             inp = int(input("Pick a row, 1-7: "))
    #             inp -= 1
    #             return inp
    #         except Exception as e:
    #             print("Invalid choice.")

    # def play_cpu(self, root_state):
    #     state = np

    # def play_cpu(self, root_state, curr_player=0):
    #     self.best.eval()

    #     root_state = np.array(root_state, dtype="float32")
    #     joint_state = [np.copy(root_state), np.copy(root_state)]
    #     results = dict()
    #     results["player_one"] = 0
    #     results["player_two"] = 0
    #     results["draw"] = 0
    #     np.set_printoptions(precision=3)

    #     game_over = False
    #     joint = np.copy(joint_state)
    #     while not game_over:
    #         legal_actions = self.get_legal_actions(joint)
    #         if len(legal_actions) == 0:
    #             results["draw"] += 1
    #             break
    #         if curr_player == 0:
    #             joint_copy = np.copy(joint)
    #             blank = [["_" for _ in range(7)] for _ in range(6)]
    #             # dsp = np.array(blank, dtype=object)
    #             m1 = np.ma.masked_where(joint_copy[0] > 0, blank)
    #             np.ma.set_fill_value(m1, "O")
    #             m1 = m1.filled()
    #             m2 = np.ma.masked_where(joint_copy[1] > 0, m1)
    #             np.ma.set_fill_value(m2, "X")
    #             m2 = m2.filled()
    #             print(m2)

    #             row = self.choose_row()
    #             idx = legal_actions[row]
    #             action = self.actions[idx]
    #         else:
    #             pi, _ = self.run_simulations(
    #                 joint, curr_player, 0)
    #             print(pi)

    #             pi = self.apply_temp_to_policy(pi, 0, T=0)

    #             idx = np.random.choice(len(self.actions), p=pi)

    #             action = self.actions[idx]

    #         joint[curr_player] = self.transition(joint[curr_player],
    #                                              action)
    #         reward, game_over = self.calculate_reward(joint)

    #         if game_over:
    #             if reward == -1:
    #                 results["player_two"] += 1
    #             elif reward == 1:
    #                 results["player_one"] += 1
    #         else:
    #             curr_player += 1
    #             curr_player = curr_player % 2

    #     print(results)
