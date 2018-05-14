from model_utils import eval_mode, setup_models, setup_optims, cast_to_torch, cast_to_cuda, cast_to_variable, train_mode
from models import SoftmaxModule
from utils import IOStream, create_folders
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd import gradcheck

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

np.seterr(all="raise")

def activate_cuda(models):
    for model in models:
        model = model.cuda()

def set_eval_mode(models):
    for model in models:
        model.eval()

def set_train_mode(models):
    for model in models:
        model.train()

class MCTSnet:
    def __init__(self,
                 actions,
                 get_legal_actions,
                 transition_and_evaluate,
                 cuda=torch.cuda.is_available(),
                 best=False):
        utils.create_folders()
        self.cuda = cuda

        self.actions = actions
        self.get_legal_actions = get_legal_actions
        self.transition_and_evaluate = transition_and_evaluate

        self.models = model_utils.load_models()
        self.best_models = model_utils.load_models()

        if self.cuda: 
            activate_cuda(self.models)
            activate_cuda(self.best_models)

    def choose_row(self):
        while True:
            try:
                inp = int(input("Pick a row, 1-7: "))
                inp -= 1
                return inp
            except Exception as e:
                print("Invalid choice.")

    def play_cpu(self, root_state, curr_player=0):
        set_eval_mode(self.models)

        root_state = np.array(root_state, dtype="float32")
        joint_state = [np.copy(root_state), np.copy(root_state)]
        results = dict()
        results["player_one"] = 0
        results["player_two"] = 0
        results["draw"] = 0
        np.set_printoptions(precision=3)

        game_over = False
        joint = np.copy(joint_state)
        while not game_over:
            legal_actions = self.get_legal_actions(joint)
            if len(legal_actions) == 0:
                results["draw"] += 1
                break
            if curr_player == 0:
                joint_copy = np.copy(joint)
                blank = [["_" for _ in range(7)] for _ in range(6)]
                # dsp = np.array(blank, dtype=object)
                m1 = np.ma.masked_where(joint_copy[0] > 0, blank)
                np.ma.set_fill_value(m1, "O")
                m1 = m1.filled()
                m2 = np.ma.masked_where(joint_copy[1] > 0, m1)
                np.ma.set_fill_value(m2, "X")
                m2 = m2.filled()
                print(m2)

                row = self.choose_row()
                idx = legal_actions[row]
                action = self.actions[idx]
            else:
                pi, _ = self.run_simulations(
                    joint, curr_player, 0)

                print(pi)

                pi = self.apply_temp_to_policy(pi, 0, T=0)

                idx = np.random.choice(len(self.actions), p=pi)

                action = self.actions[idx]

            joint[curr_player] = self.transition(joint[curr_player],
                                                 action)
            reward, game_over = self.calculate_reward(joint)

            if game_over:
                if reward == -1:
                    results["player_two"] += 1
                elif reward == 1:
                    results["player_one"] += 1
            else:
                curr_player += 1
                curr_player = curr_player % 2

        print(results)

    def do_round(self, results, joint_state, curr_player, T=config.TAU, record_memories=True):
        if record_memories:
            memories = []
        game_over = False
        joint = np.copy(joint_state)
        turn = 0
        while not game_over:
            turn += 1
            legal_actions = self.get_legal_actions(joint)
            if len(legal_actions) == 0:
                results["draw"] += 1
                break
            if curr_player == 0:
                pi, memory = self.run_simulations(joint, curr_player, turn)
            else:
                pi, memory = self.best_net.run_simulations(
                    joint, curr_player, turn)

            pre_temp_idx = np.random.choice(len(self.actions), p=pi)
            pi = self.apply_temp_to_policy(pi, turn, T)
            idx = np.random.choice(len(self.actions), p=pi)

            memory["readout"]["output"] = F.log_softmax(
                memory["readout"]["output"], dim=0)[pre_temp_idx]

            if record_memories:
                memories.extend([memory])

            action = self.actions[idx]

            joint[curr_player] = self.transition(joint[curr_player],
                                                 action)
            reward, game_over = self.calculate_reward(joint)

            if game_over:
                if reward == -1:
                    results["player_two"] += 1
                elif reward == 1:
                    results["player_one"] += 1
            else:
                curr_player += 1
                curr_player = curr_player % 2
        if record_memories:
            for memory in memories:
                if memory["curr_player"] == 0:
                    memory["result"] = reward
                else:
                    memory["result"] = -1*reward
            self.memories.extend(memories)

    def self_play(self, root_state, curr_player=0, save_model=True,
                  T=config.TAU, record_memories=True):
        eval_mode(self.models)

        root_state = np.array(root_state, dtype="float32")
        joint_state = [np.copy(root_state), np.copy(root_state)]
        results = dict()
        results["player_one"] = 0
        results["player_two"] = 0
        results["draw"] = 0
        np.set_printoptions(precision=3)

        for _ in tqdm(range(config.EPISODES)):
            self.do_round(results, joint_state, curr_player, T=T, record_memories=record_memories)

        # results["player_one"] = 0
        # results["player_two"] = 0
        # results["draw"] = 0
        # for _ in tqdm(range(config.EVALUATION_EPISODES)):
        #     self.do_round(results, joint_state, curr_player,
        #                   T=0, record_memories=False)
        # print("Deterministic Results: ", results)
        if T==0:
            name="Deterministic"
        else:
            name="Stochastic"
        print("{} Results: ".format(name), results)
        if save_model:
            if results["player_one"] > results["player_two"] * config.SCORING_THRESHOLD:
                self.save_best_model()
                self.best_net.models = setup_models(
                    self.best_net.io, self.best_net.load_model, self.best_net.cuda, trainer=False)
                self.best_net.optims = setup_optims(
                    self.best_net.models, self.best_net.cuda)

            elif results["player_two"] > results["player_one"] * config.SCORING_THRESHOLD:
                # load best model to training model
                self.models = setup_models(
                    self.io, self.load_model, self.cuda, trainer=False)
                self.optims = setup_optims(self.models, self.cuda)

            # self.save_training_model()
        # self.memories = self.memories[-config.MAX_MEMORIES:]
        print("Num memories: {}".format(len(self.memories)))
        # Note, I am loading old memories from a bad version
        # It will eventually get overwritten, but it is a little inefficient to reference those
        return self.memories

    def save_best_model(self):
        self.io.cprint("Saving best model")
        for name, model in self.models.items():
            torch.save(model, "checkpoints/models/%s.t7" % (name + "_best"))

    def save_training_model(self):
        self.io.cprint("Saving training model")
        for name, model in self.models.items():
            torch.save(model, "checkpoints/models/%s.t7" %
                       (name + "_training"))

    def load_training_model(self):
        self.models = setup_models(
            self.io, self.load_model, self.cuda, trainer=True)
        self.optims = setup_optims(self.models, self.cuda)

    def save_memories(self):
        print("Saving Memories...")
        pickle.dump(self.memories, open("checkpoints/memories.p", "wb"))

    def plot_losses(self):
        plt.plot(self.history["readout"], "r")
        plt.plot(self.history["policy"], "m")
        plt.plot(self.history["value"], "c")
        plt.plot(self.history["total"], "y")
        plt.show()

    def run_simulations(self, joint_states, curr_player, turn):
        self.embeddings = dict()
        S = dict()
        A = dict()
        R = dict()
        H = dict()
        N = dict()
        game_over = False
        memory = {
            "curr_player": curr_player,
            "result": None,
            "policy": {
                "output": []
            },
            "readout": {
                "output": None
            },
            "value": {
                "output": None
            }
        }

        root_state = np.concatenate((np.expand_dims(joint_states[0], 0),
                               np.expand_dims(joint_states[1], 0),
                               np.zeros(shape=np.expand_dims(joint_states[1], 0).shape) + curr_player), axis=0)

        t = 0
        #+1 sims since the first is used to expand the embedding
        for sim in range(config.MCTS_SIMS+1):
            while True:
                try:
                    N[hashed] += 1
                except:
                    N[hashed] = 0
                    break

                legal_actions = self.get_legal_actions(S[t][:2])

                reward, game_over = self.calculate_reward(S[t][:2])

                R[t] = reward
                if len(legal_actions) == 0 or game_over:
                    game_over = True
                    break

                # consider moving the value head here and using it in the backups
                action = self.simulate(self.embeddings[hashed], S[t],
                                       sim, memory)

                A[t] = action

                new_state = self.transition(
                    np.copy(S[t][:2][curr_player]), A[t])
                S[t + 1] = np.copy(S[t])
                S[t + 1][curr_player] = np.copy(new_state)
                t += 1
                curr_player += 1
                curr_player = curr_player % 2
                S[t][2] = curr_player
                S[t].flags.writeable = False
                hashed = hash(S[t].data.tobytes())
                S[t].flags.writeable = True

            if not game_over and len(legal_actions) > 0:
                state_one = cast_to_torch(S[t][0], self.cuda).unsqueeze(0)
                state_two = cast_to_torch(S[t][1], self.cuda).unsqueeze(0)
                state_three = cast_to_torch(S[t][2], self.cuda).unsqueeze(0)
                state = torch.cat(
                    [state_one, state_two, state_three], 0).unsqueeze(0)
                self.models["emb"].eval()
                H[t] = self.embeddings[hashed] = self.models["emb"](state)

            if t > 0:
                H = self.backup(H, R, S, t, memory)
                t = 0

        self.models["readout"].eval()

        logits = self.models["readout"](H[0])

        memory["readout"]["output"] = logits

        pi = self.correct_policy(logits, joint_states, is_root=False)

        return pi, memory

    def apply_temp_to_policy(self, pi, turn, T=config.TAU):
        if turn == config.TURNS_UNTIL_TAU0 or T == 0:
            temp = np.zeros(shape=pi.shape)
            temp[np.argmax(pi)] = 1
            pi = temp

        return pi

    def simulate(self, emb, joint_state, sim, memory):
        emb = emb.view(1, 1, 8, 16)
        self.models["policy"].eval()
        logits, value = self.models["policy"](emb)

        if sim == 1:
            is_root = True
        else:
            is_root = False
        pi = self.correct_policy(logits, joint_state, is_root=is_root)

        idx = np.random.choice(len(self.actions), p=pi)

        action = self.actions[idx]
        memory["policy"]["output"].append({
            "log_action_prob": F.log_softmax(logits, dim=0)[idx], 
            "value": value, 
            "is_root": is_root
        })

        return action

    def backup(self, H, R, S, _t, memory, is_for_inp=False):
        for t in reversed(range(_t)):
            reward = cast_to_torch([R[t]], self.cuda)
            comb_state_1 = S[t + 1][0] + S[t + 1][1]
            comb_state_2 = S[t][0] + S[t][1]
            action = comb_state_1 - comb_state_2
            action = cast_to_torch(action, self.cuda).view(-1)

            inp = torch.cat([H[t], H[t + 1], reward, action], 0)

            self.models["backup"].eval()
            H[t] = self.models["backup"](inp, H[t])

        return H

    def correct_policy(self, logits, joint_state, is_root):
        odds = np.exp(logits.data.numpy())
        policy = odds / np.sum(odds)
        if is_root:
            nu = np.random.dirichlet([config.ALPHA] * len(self.actions))
            policy = policy * (1 - config.EPSILON) + nu * config.EPSILON

        mask = np.zeros(policy.shape)
        legal_actions = self.get_legal_actions(joint_state[:2])
        mask[legal_actions] = 1
        policy = policy * mask

        pol_sum = (np.sum(policy) * 1.0)
        if pol_sum == 0:
            return policy
        else:
            return policy / pol_sum

        return policy

    def zero_grad(self):
        for _, optim in self.optims.items():
            optim.zero_grad()

    def optim_step(self):
        for _, optim in self.optims.items():
            optim.step()

    def train(self, minibatches, last_loop=False):
        for e in range(config.EPOCHS):
            last_epoch = (e == (config.EPOCHS-1))
            if e > 0:
                shuffle(minibatches)
            read_loss_data = 0
            pol_loss_data = 0
            val_loss_data = 0
            total_loss_data = 0

            for mb in minibatches:
                self.zero_grad()

                pol_loss = 0
                val_loss = 0
                read_loss = 0
                weights = [1, 1, 1]

                num_val_losses = 0
                num_pol_losses = 0
                num_read_losses = 0

                for i, memory in enumerate(mb):
                    result = memory["result"]
                    pol_trajectories = memory["policy"]["output"]

                    for action in pol_trajectories:
                        if action["is_root"]:
                            root_value = action["value"]
                            root_log_action_prob = action["log_action_prob"]
                        else:
                            pol_loss += - \
                                action["log_action_prob"] * \
                                (result - action["value"])
                            num_pol_losses += 1
                            val_loss += F.mse_loss(action["value"],
                                                   Variable(torch.FloatTensor(np.array([result])), volatile=True))
                            num_val_losses += 1
                    val_loss += F.mse_loss(root_value,
                                           Variable(torch.FloatTensor(np.array([result])), volatile=True))
                    pol_loss += -root_log_action_prob*(result - root_value)
                    read_loss += - \
                        memory["readout"]["output"]*(result - root_value)
                    num_pol_losses += 1
                    num_val_losses += 1
                    num_read_losses += 1
                val_loss = val_loss / (len(mb))
                pol_loss = pol_loss / (len(mb))
                read_loss = read_loss / (len(mb))

                total_loss = (read_loss * weights[0] + pol_loss *
                              weights[1] + val_loss*weights[2])

                read_loss_data += read_loss.data.numpy()[
                    0]*weights[0]
                pol_loss_data += pol_loss.data.numpy()[
                    0]*weights[1]
                val_loss_data += val_loss.data.numpy()[0]*weights[2]
                total_loss_data += total_loss.data.numpy()[0]

                # if (last_epoch):
                #     total_loss.backward(retain_graph=False)
                # else:
                #     total_loss.backward(retain_graph=True)
                total_loss.backward()
                assert (root_value.grad is not None and root_log_action_prob.grad is not None and
                        mb[0]["readout"]["output"].grad is not None)
                set_trace()

                # orig_params = {}
                # for name, model in self.models.items():
                #     orig_params[name] = []
                #     for parameters in model.parameters():
                #         orig_params[name].extend([np.copy(parameters.detach().data.numpy())])

                self.optim_step()
                # for name, model in self.models.items():
                #     for i, parameters in enumerate(model.parameters()):
                #         if not (orig_params[name][i] == parameters.detach().data.numpy()).all():
                #             print(name)
                #             set_trace()
                #             test = "hi"

            read_loss_data /= len(minibatches)
            pol_loss_data /= len(minibatches)
            val_loss_data /= len(minibatches)
            total_loss_data /= len(minibatches)

            if len(self.history["readout"]) == 0:
                self.history["readout"].extend([read_loss_data])
                self.history["policy"].extend([pol_loss_data])
                self.history["value"].extend([val_loss_data])
                self.history["total"].extend([total_loss_data])
                pickle.dump(self.history, open("checkpoints/history.p", "wb"))

            elif last_loop and last_epoch and len(self.history["readout"]) > 0:
                prev_readout = self.history["readout"][-1]
                prev_policy = self.history["policy"][-1]
                prev_value = self.history["value"][-1]
                prev_total = self.history["total"][-1]
                r_sign = "" if prev_readout > read_loss_data else "+"
                p_sign = "" if prev_policy > pol_loss_data else "+"
                v_sign = "" if prev_value > val_loss_data else "+"
                t_sign = "" if prev_total > total_loss_data else "+"
                r_diff = ((read_loss_data -
                           prev_readout) / prev_readout)*100
                p_diff = ((pol_loss_data -
                           prev_policy) / prev_policy)*100
                v_diff = ((val_loss_data - prev_value) / prev_value)*100
                t_diff = ((total_loss_data - prev_total) / prev_total)*100

                print("readout loss: {} ({}{}%)"
                      .format(np.round(read_loss_data, 4), r_sign, r_diff))
                print("policy loss: {} ({}{}%)"
                      .format(np.round(pol_loss_data, 4), p_sign, p_diff))
                print("value loss: {} ({}{}%)"
                      .format(np.round(val_loss_data, 4), v_sign, v_diff))
                print("total loss: {} ({}{}%)"
                      .format(np.round(total_loss_data, 4), t_sign, t_diff))

                self.history["readout"].extend([read_loss_data])
                self.history["policy"].extend([pol_loss_data])
                self.history["value"].extend([val_loss_data])
                self.history["total"].extend([total_loss_data])
                pickle.dump(self.history, open("checkpoints/history.p", "wb"))

    def train_memories(self):
        train_mode(self.models)
        self.io.cprint("Training memories")

        # add a test there that takes the oldest memories, creates a minibatches with them
        # and runs one test that sees to see if the loss is changing all of the parts
        # of the network, i.e. the parameters before and after the update are different

        # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
        # gives some hints. param update magnitudes should be 1e-3
        # if len(self.memories) > config.MIN_MEMORIES:
        # num_samples = config.NUM_SAMPLES - (config.NUM_SAMPLES%config.BATCH_SIZE)
        

        for i in tqdm(range(config.TRAINING_LOOPS)):
            last_loop = (i == (config.TRAINING_LOOPS-1))
            shuffle(self.memories)
            minibatches = [self.memories[:config.BATCH_SIZE]]
            # minibatches = [
            #     data[x:x + config.BATCH_SIZE]
            #     for x in range(0, len(data), config.BATCH_SIZE)
            # ]
            self.train(minibatches, last_loop)
        # else:
        #     print("Not enough memories to train, resuming self-play.")
