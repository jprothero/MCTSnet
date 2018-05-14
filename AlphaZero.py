import numpy as np
from ipdb import set_trace

class AlphaZero:
    def __init__(self, turns_until_tau0=10, alpha=.8, epsilon=.2, c=1):
        self.turns_until_tau0 = turns_until_tau0
        self.alpha = alpha
        self.epsilon = epsilon
        self.c = c

        self.reset()

    def reset(self):
        self.curr_node = {
            "children": None,
            "parent": None,
            "N": 0,
            "d": 0
        }

        self.turn = 1

        self.T = 1

    def select(self, state, transition_and_evaluate):
        sim_over = False
        result = None

        while self.curr_node["children"] is not None and not sim_over:
            choice_idx = self.curr_node["max_uct_idx"]
            self.curr_node = self.curr_node["children"][choice_idx]
            state, result, sim_over = transition_and_evaluate(state, choice_idx)

        return state, result, sim_over

    def select_real(self):
        visits = np.array([child["N"] for child in self.curr_node["children"]])

        if self.turn == self.turns_until_tau0:
            self.T = 0

        if self.T != 0:
            visits_sum = (1.0 * visits.sum())

            if visits_sum == 0:
                print("WARNING: visits_sum is 0")
                idx = np.argmax(visits)
            else:
                visits = visits / visits_sum
                idx = np.random.choice(len(visits), p=visits)
        else:
            idx = np.argmax(visits)

        self.turn += 1

        self.curr_node = self.curr_node["children"][idx]
        self.curr_node["parent"] = None

        return idx, visits

    def expand(self, policy):
        self.curr_node["children"] = []

        if self.curr_node["parent"] is None:
            policy = self.add_dirichlet_noise(policy)

        for p in policy:
            child = {
                "N": 0,
                "W": 0,
                "Q": 0,
                "U": p,
                "P": p,
                "d": self.curr_node["d"]+1,
                "children": None,
                "parent": self.curr_node
            }

            self.curr_node["children"].extend([child])

        return self.curr_node

    def update_uct(self):
        if self.curr_node["children"] is not None:
            self.curr_node["max_uct"] = -1
            self.curr_node["max_uct_idx"] = -1
            for i, child in enumerate(self.curr_node["children"]):
                child["U"] = self.c*child["P"] * \
                    (1 + np.log(self.curr_node["N"])/(1 + child["N"]))
                child["UCT"] = child["Q"] + child["U"]
                if child["UCT"] > self.curr_node["max_uct"]:
                    self.curr_node["max_uct"] = child["UCT"]
                    self.curr_node["max_uct_idx"] = i

    def backup(self, value):
        value += 1
        value /= 2
        
        while self.curr_node["parent"] is not None:
            self.update_node(value)
            
            self.update_uct()
            
            self.curr_node = self.curr_node["parent"]

        #update root visits
        self.curr_node["N"] += 1

        self.update_uct()        

    def update_node(self, value):
        self.curr_node["N"] += 1
        self.curr_node["W"] += value
        self.curr_node["Q"] = self.curr_node["W"]/self.curr_node["N"]

    def add_dirichlet_noise(self, policy):
        alpha = min(10/len(policy), .8)
        epsilon = 0.25
        nu = np.random.dirichlet([alpha] * len(policy))
        policy = policy*(1-epsilon) + nu*epsilon

        return policy

