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
        self._create_legal_moves_pattern()
        self._create_actions()
        
    def _create_legal_moves_pattern(self):
#         self.legal_move_pattern = np.array([1 for i in range(self.rows)])
        self.legal_move_pattern = np.array([0])
        self.legal_move_pattern = np.expand_dims(self.legal_move_pattern, -1)
        
    # def calculate_reward(self, joint_states, idx):
    #     for k, state in enumerate(joint_states):
    #         match = cv2m(pattern.astype(self.datatype), state.astype(self.datatype), 
    #                         cv2.TM_SQDIFF)

    #         i, j = np.where(match==0)
    #         if len(i) != 0 or len(j) != 0:
    #             if k == 0:
    #                 return 1, True
    #             else:
    #                 return -1, True

    #     return 0, False
    
    def _create_actions(self):
        self.actions = [i for i in range(self.rows*self.columns)]

    def get_legal_actions(self, joint_states):
        assert len(joint_states) == 2
        board = joint_states[0] + joint_states[1]
        
        legal_moves = []
        
        for k in range(board.shape[1]):
            match = cv2m(self.legal_move_pattern.astype(self.datatype), 
                     board[:, k].astype(self.datatype), cv2.TM_SQDIFF)
            
            i, j = np.where(match==0)
            
            if len(i) != 0:
                legal_moves.append(np.max(i)*board.shape[1] + k)

        return legal_moves

    def transition_and_evaluate(self, full_state, action):
        orig_player = int(full_state[2][0][0])

        plane = full_state[orig_player]

        idx = action
        
        i, j = np.unravel_index([idx], plane.shape)
        i = i[0]
        j = j[0]
        
        plane[i][j] = 1

        new_player = (orig_player+1)%2
        full_state[2] = new_player
        #right diag and up down are working
        #left right is working
        # assert full_state[orig_player][i][j] == 1
        game_over = check_win(plane, i, j)

        if game_over:
            result = 1
        else:
            legal_actions = self.get_legal_actions(full_state[:2])
            if len(legal_actions) == 0:
                result = 0
                game_over = True
            else:
                result = None

        return full_state, result, game_over

#### Static testing functions
# def test_transition():
#     connect4 = Connect4()
    
#     board = np.zeros(shape=(6, 7))
    
#     for i in range(1000):
#         action = np.random.randint(42)
        
#         test = np.copy(board.flatten())
#         test[action] = 1

#         board = connect4.transition(board, action)
        
#         assert (board.flatten() == test).all()

def test_legal_moves_finder():
    connect4 = Connect4()

    board = np.zeros(shape=(6, 7))
    res = connect4.get_legal_actions(board)
    assert len(res) == 7 and res[0] == 35
    board[1:, :] = 1
    res = connect4.get_legal_actions(board)
    assert len(res) == 7 and res[0] == 0
    board[0][0] = 1
    res = connect4.get_legal_actions(board)
    assert len(res) == 6

def test_win_checkers():
    board = np.zeros((6, 7))
    
    left_win = np.copy(board)
    right_win = np.copy(board)
    up_win = np.copy(board)
    down_win = np.copy(board)
    left_up_diag_win = np.copy(board)
    right_up_diag_win = np.copy(board)
    left_down_diag_win = np.copy(board)
    right_down_diag_win = np.copy(board)
    
    left_win[0][:4] = 1; left_win #0, 3
    assert check_win(left_win, 0, 3)
    left_win[0][0] = 0
    assert not check_win(left_win, 0, 3)
    
    right_win[0][:4] = 1; right_win #0, 0
    assert(check_win(right_win, 0, 0))
    right_win[0][3] = 0
    assert not check_win(right_win, 0, 0)
    
    for i in range(4):
        up_win[i][0] = 1
    assert(check_win(up_win, 3, 0))
    up_win[0][0] = 0
    assert not check_win(up_win, 3, 0)
    
    for i in range(4):
        down_win[i][0] = 1
    assert(check_win(down_win, 0, 0))
    down_win[3][0] = 0
    assert not check_win(down_win, 0, 0)
    
    for i in range(4):
        left_up_diag_win[i, i] = 1
    assert(check_win(left_up_diag_win, 3, 3))
    left_up_diag_win[0][0] = 0
    assert not check_win(left_up_diag_win, 3, 3)
    
    for i in range(4):
        left_down_diag_win[i, 3-i] = 1
    assert(check_win(left_down_diag_win, 0, 3))
    left_down_diag_win[3][0] = 0
    assert not check_win(left_down_diag_win, 0, 3)
    
    for i in range(4):
        right_up_diag_win[3-i, i] = 1
    assert(check_win(right_up_diag_win, 3, 0))
    right_up_diag_win[0][3] = 0
    assert not check_win(right_up_diag_win, 3, 0)
    
    for i in range(4):
        right_down_diag_win[i, i] = 1
    assert check_win(right_down_diag_win, 0, 0)
    right_down_diag_win[3][3] = 0
    assert not check_win(right_down_diag_win, 0, 0)

def check_win(state, i, j):
    done = False
    done = check_up_down(state, i, j)
    if done:
        return done
    done = check_left_right(state, i, j)
    if done:
        return done
    done = check_right_diag(state, i, j)
    if done:
        return done
    return check_left_diag(state, i, j)

def check_up_down(state, i, j):
    num_in_a_row = 0
    for r in range(state.shape[0]):
        if state[r, j] == 0:
            num_in_a_row = 0
        else:
            num_in_a_row += 1
            if num_in_a_row == 4:
                return True
    return False

def check_left_right(state, i, j):
    num_in_a_row = 0
    for c in range(state.shape[1]):
        if state[i, c] == 0:
            num_in_a_row = 0
        else:
            num_in_a_row += 1
            if num_in_a_row == 4:
                return True
    return False

def check_right_diag(state, i, j):
    r = i
    c = j
    while r < state.shape[0]-1 and c > 0:
        r += 1
        c -= 1
    
    if c == 0:
        num_in_a_row = 0
        for c in range(state.shape[1]):
            if r < 0:
                break
            if state[r, c] == 0:
                num_in_a_row = 0
            else:
                num_in_a_row += 1
                if num_in_a_row == 4:
                    return True
            r -= 1
    else:
        num_in_a_row = 0
        for r in reversed(range(state.shape[0])):
            if c > state.shape[1]-1:
                break
            if state[r, c] == 0:
                num_in_a_row = 0
            else:
                num_in_a_row += 1
                if num_in_a_row == 4:
                    return True 
                
            c += 1
            
    return False

def check_left_diag(state, i, j):
    r = i
    c = j
    while r > 0 and c > 0:
        r -= 1
        c -= 1
    
    if c == 0:
        num_in_a_row = 0
        for c in range(state.shape[1]):
            if r > state.shape[0]-1:
                break
            if state[r, c] == 0:
                num_in_a_row = 0
            else:
                num_in_a_row += 1
                if num_in_a_row == 4:
                    return True
            r += 1
    else:
        num_in_a_row = 0
        for r in range(state.shape[0]):
            if c > state.shape[1]-1:
                break
            if state[r, c] == 0:
                num_in_a_row = 0
            else:
                num_in_a_row += 1
                if num_in_a_row == 4:
                    return True 
                
            c += 1
            
    return False