import cv2
from cv2 import matchTemplate as cv2m
import numpy as np
from IPython.core.debugger import set_trace
import torch.functional as F

class Connect4:
    def __init__(self, rows=6, columns=7, datatype="uint8"):
        self.rows = rows
        self.columns = columns
        self.datatype = datatype
        self._create_win_patterns()
        self._create_legal_moves_pattern()
        self._create_valid_pattern()
        self._create_valid_pattern2()
        self._create_actions()
        
    def _create_win_patterns(self):
        self.win_patterns = [i for i in range(4)]
        self.horizontal_win = self.win_patterns[0] = np.array([[1, 1, 1, 1]])
        
        vertical_win = np.array([1, 1, 1, 1])
        vertical_win = np.expand_dims(vertical_win, -1)
        self.vertical_win = self.win_patterns[1] = vertical_win
        
        self.left_diag_win = self.win_patterns[2] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.right_diag_win = self.win_patterns[3] = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
        ])
        
    def _create_legal_moves_pattern(self):
#         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.legal_move_pattern = np.array([0])
        self.legal_move_pattern = np.expand_dims(self.legal_move_pattern, -1)

    def _create_valid_pattern(self):
#         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.valid_pattern = np.array([1, 0])
        self.valid_pattern = np.expand_dims(self.valid_pattern, -1)

    def _create_valid_pattern2(self):
    #         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.valid_pattern2 = np.array([2])
        self.valid_pattern2 = np.expand_dims(self.valid_pattern2, -1)
        
    def calculate_reward(self, joint_states):
        for k, state in enumerate(joint_states):
            for pattern in self.win_patterns:
                match = cv2m(pattern.astype(self.datatype), state.astype(self.datatype), 
                             cv2.TM_SQDIFF)

                i, j = np.where(match==0)
                if len(i) != 0 or len(j) != 0:
                    if k == 0:
                        return 1, True
                    else:
                        return -1, True

        return 0, False
    
    def _create_actions(self):
        self.actions = [i for i in range(self.rows*self.columns)]

    def get_legal_actions(self, joint_states):
        assert len(joint_states) == 2
        board = joint_states[0] + joint_states[1]
        
        legal_moves = []
        
        for k in range(board.shape[1]):
            match = cv2m(self.legal_move_pattern.astype(self.datatype), 
                     board[:, k].astype(self.datatype), cv2.TM_SQDIFF);
            
            i, j = np.where(match==0)
            
            if len(i) != 0:
                legal_moves.append(np.max(i)*board.shape[1] + k)
                
#         print(board)
#         if len(board.shape) > 2:
#         print(board)
#         print(legal_moves)
#         set_trace()
#         pass
        return legal_moves

    def test_valid(self, state):
        board = state[0] + state[1]
        
        for k in range(board.shape[1]):
            match = cv2m(self.valid_pattern2.astype(self.datatype), 
                     board[:, k].astype(self.datatype), cv2.TM_SQDIFF)
            
            i, j = np.where(match==0)

            if len(i) > 0:
                return -1

            match = cv2m(self.valid_pattern.astype(self.datatype), 
                     board[:, k].astype(self.datatype), cv2.TM_SQDIFF)
            
            i, j = np.where(match==0)
            
            #so I'm trying to make sure that there are no holes
            #i.e. we want t
            if len(i) > 0:
                return -1
            
        return 1     
#         print(board)
#         if len(board.shape) > 2:
#         print(board)
#         print(legal_moves)
#         set_trace()
#         pass
    
    def transition(self, state, action, debug=False):
        if debug:
            set_trace()
        idx = action
        
        i, j = np.unravel_index([idx], state.shape)
        
        state[i[0]][j[0]] = 1
        return state

    def get_legal_mask(self, input_states, add_noise=False, deterministic=False):
            legal_moves_mask = np.copy(input_states.data.numpy())

            log_probas_list = []
            legal_moves_lists = []
            
            #legal_moves_mask will be (128,3 , 6, 7)
            #I only care about the first two states, so that's right
            for s, state in enumerate(legal_moves_mask[:, :2]):
                state_legal_moves_list = []
                for k in range(state.shape[1]):
                    match = cv2m(self.legal_move_pattern.astype(self.datatype), 
                                state[:, k].astype(self.datatype), cv2.TM_SQDIFF)
                    
                    i, j = np.where(match==0)
                    
                    if len(i) != 0:
                        #I think I need a k index
                        idx = np.max(i)*state.shape[1] + k
                        legal_moves_mask[s][k][idx] = 1
                        state_legal_moves_list.extend([idx])

                legal_moves_list.extend(state_legal_moves_list)
                legal_moves_mask[s][:2].flatten()[state_legal_moves_list] = 1

            return legal_moves_mask, legal_moves_list

    # def make_valid_transition(self, new_states, legal_moves_list, add_noise=False, deterministic=False):
    #     log_probas_list = []
    #     draws = []
    #     for s, state in enumerate(new_states[:, :2]):
    #         state_legal_moves = legal_moves_list[s]

    #         if len(state_legal_moves) == 0:
    #             draws.extend([s])
    #             continue

    #         view = state.flatten()[state_legal_moves]
    #         if add_noise and not deterministic:
    #             nu = np.random.dirichlet([config.ALPHA] * len(state_legal_moves))
    #             view = view * (1 - config.EPSILON) + nu * config.EPSILON
    #         probas = F.softmax(view, dim=0)
    #         if deterministic:
    #             probas_idx = np.argmax(probas.data.numpy()))
    #         else:
    #             probas_idx = np.random.choice(len(probas.data.numpy()), p=probas)
    #         log_probas = F.log_softmax(view, dim=0)
    #         log_probas_list.extend([log_probas[probas_idx]])

    #         idx = probas.data.numpy()[probas_idx]

    #         view = 0
    #         batch_of_states[s].flatten()[idx] = 1

    #     return new_states, log_probas_list, draws

def test_transition():
    connect4 = Connect4()
    
    board = np.zeros(shape=(6, 7))
    
    for i in range(1000):
        action = np.random.randint(42)
        
        test = np.copy(board.flatten())
        test[action] = 1

        board = connect4.transition(board, action)
        
        assert (board.flatten() == test).all()
    
def test_win_finder():
    connect4 = Connect4()

    board = np.zeros(shape=(6, 7))

    assert connect4.check_win(board) == 0

    # #horizontal win
    board[0][0:4] = 1; 

    assert connect4.check_win(board) != 0

    board = np.zeros(shape=(6, 7))

    #vertical win
    board[0][0] = 1
    board[1][0] = 1
    board[2][0] = 1
    board[3][0] = 1

    assert connect4.check_win(board) != 0

    board = np.zeros(shape=(6, 7))

    #left diag win
    for i in range(4):
        board[i][i] = 1; 

    assert connect4.check_win(board) != 0

    board = np.zeros(shape=(6, 7))

    #right diag win
    for i in range(4):
        board[len(board[0])-i-2, i] = 1

    assert connect4.check_win(board) != 0

    board = np.zeros(shape=(6, 7))

# test_win_finder()

def test_legal_moves_finder():
    connect4 = Connect4()

    board = np.zeros(shape=(6, 7))
    res = connect4.get_legal_moves(board)
    assert len(res) == 7 and res[0] == 35
    board[1:, :] = 1
    res = connect4.get_legal_moves(board)
    assert len(res) == 7 and res[0] == 0
    board[0][0] = 1
    res = connect4.get_legal_moves(board)
    assert len(res) == 6

# test_legal_moves_finder()