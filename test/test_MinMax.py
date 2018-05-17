import unittest
from MinMax import evaluateScore, bestMove, COMPUTER_PLAYER, HUMAN_PLAYER

# FUncion para detectar victoria debe ser diferente a la que
# evalue en minimax

# La que evalua victoria solo analiza el tablero y regresa lo
# primero que encuentra

# Y la de minimax elige la victoria o derrota dependiendo de si es su turno
# o no. Si tiene oportunidad de ganar y puede hacerlo, que lo haga. Si va
# a perder inevitablemente, que lo detenga

class TestHeuristic(unittest.TestCase):

    def setUp(self):
        self.player = COMPUTER_PLAYER
        self.opponent = HUMAN_PLAYER

    def test_score_potential_wins(self):
        gameState = [
            [ 1, 0,  0,  0, 0, 0, 0],
            [-1, 0,  0,  0, 0, 0, 0],
            [-1, 0,  1, -1, 0, 0, 0],
            [-1, 0, -1,  1, 0, 0, 0],
            [ 1, 0,  1,  1, 0, 0, 0],
            [-1, 0, -1, -1, 0, 0, 0]
        ]
        score = evaluateScore(gameState, self.player, self.opponent)
        self.assertEqual(score, 0)

        gameState = [
            [ 1, 0,  0,  0,  0, 0, 0],
            [-1, 0,  0,  0,  0, 0, 0],
            [ 1, 0,  1,  1,  1, 0, 0],
            [-1, 0, -1, -1,  1, 0, 1],
            [-1, 0, -1, -1,  1, 0, 1],
            [-1, 0, -1, -1, -1, 0, 1]
        ]
        score = evaluateScore(gameState, self.player, self.opponent)
        self.assertEqual(score, 0)

    def test_score_computer_win(self):
        gameState = [
            [-1,  0,  0,  0, 0, 0, 0],
            [-1,  0,  0,  0, 0, 0, 0],
            [ 1,  0,  0,  0, 0, 0, 0],
            [-1,  1,  0,  0, 0, 0, 0],
            [ 1,  1,  1,  0, 0, 0, 0],
            [-1, -1, -1,  1, 0, 0, 0]
        ]
        score = evaluateScore(gameState, self.player, self.opponent)
        self.assertEqual(score, float("inf"))

    def test_score_opponent_win(self):
        gameState = [
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 1],
            [-1, 0, 0, 0, 0, 0, 1],
            [-1, 0, 0, 0, 0, 0, 1]
        ]
        score = evaluateScore(gameState, self.player, self.opponent)
        self.assertEqual(score, float("-inf"))

        gameState = [
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 1],
            [ 0, 0, 0, 0, 0, 0, 1],
            [-1,-1,-1,-1, 0, 0, 1]
        ]
        score = evaluateScore(gameState, self.player, self.opponent)
        self.assertEqual(score, float("-inf"))


class TestMinimax(unittest.TestCase):

    def setUp(self):
        self.player = COMPUTER_PLAYER
        self.opponent = HUMAN_PLAYER

    def test_prevent_defeat(self):

        gameState = [
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0,-1],
            [ 1, 0, 0, 0, 0, 0,-1],
            [ 1, 0, 0, 0, 0, 0,-1]
        ]
        move = bestMove(gameState, self.player, self.opponent)
        self.assertEqual(move, 6)

    def test_go_for_win(self):

        gameState = [
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 1, 0, 0, 0, 0,-1],
            [ 0, 1, 0, 0, 0, 0,-1],
            [ 0, 1, 0, 0, 0,-1,-1]
        ]
        move = bestMove(gameState, self.player, self.opponent)
        self.assertEqual(move, 1)
