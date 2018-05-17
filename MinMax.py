# Credit: https://github.com/nloyolag/connect4-ai

from __future__ import print_function
from __future__ import division
raw_input = input

import sys
import numpy
import unittest

BOARD_SIZE_X = 7 # col
BOARD_SIZE_Y = 6 # row
SEARCH_DEPTH = 4

COMPUTER_PLAYER = 1 # minmax
HUMAN_PLAYER = -1 # other agent


#
# Method that translate full state (3, 6, 7)
# to game state (6,7) where minmax agent holds 1 and the other agent holds -1
#
def fullState_2_gameState(full_state, minmax_agent_idx: int):
    # assert right game board shape
    assert full_state[minmax_agent_idx].shape == (BOARD_SIZE_Y, BOARD_SIZE_X)
    other_agent_idx = (minmax_agent_idx+1)%2
    # assert legal game board state
    assert ((full_state[minmax_agent_idx]+full_state[other_agent_idx]) < 2.).all()
    gameState = full_state[minmax_agent_idx]*COMPUTER_PLAYER + full_state[other_agent_idx]*HUMAN_PLAYER
    return gameState

def gameState_2_fullState(gameState, full_state, minmax_agent_idx: int):
    # assert right game board shape
    assert full_state[minmax_agent_idx].shape == (BOARD_SIZE_Y, BOARD_SIZE_X)
    other_agent_idx = (minmax_agent_idx+1)%2
    full_state[minmax_agent_idx] = np.where(gameState == COMPUTER_PLAYER, gameState, 0)
    full_state[other_agent_idx] = np.where(gameState == HUMAN_PLAYER, gameState, 0)
    # assert legal game board state
    assert ((full_state[minmax_agent_idx]+full_state[other_agent_idx]) < 2.).all()
    return

#
# Method that runs the minimax algorithm and returns
# the move and score of each call.
#

def minimax(gameState, depth, player, opponent):
    availableMoves = BOARD_SIZE_X
    for i in range(0, BOARD_SIZE_X):
        if gameState[0][i] != 0:
            availableMoves -= 1

    if depth == 0 or availableMoves == 0:
        score = evaluateScore(gameState, player, opponent)
        return None, score

    bestScore = None
    bestMove = None

    for i in range(0, BOARD_SIZE_X):
        # If moves cannot be made on column, skip it
        if gameState[0][i] != 0:
            continue

        currentMove = [0, i]

        for j in range(0, BOARD_SIZE_Y - 1):
            if gameState[j + 1][i] != 0:
                gameState[j][i] = player
                currentMove[0] = j
                break
            elif j == BOARD_SIZE_Y - 2:
                gameState[j+1][i] = player
                currentMove[0] = j+1# assert legal game board state

        # Recursive minimax call, with reduced depth
        move, score = minimax(gameState, depth - 1, opponent, player)

        gameState[currentMove[0]][currentMove[1]] = 0

        if player == COMPUTER_PLAYER:
            if bestScore == None or score > bestScore:# assert legal game board state
                bestScore = score
                bestMove = currentMove
        else:
            if bestScore == None or score < bestScore:
                bestScore = score
                bestMove = currentMove

    return bestMove, bestScore

#
# Method that calculates the heuristic value of a given
# board state. The heuristic adds a point to a player
# for each empty slot that could grant a player victory.
#

def evaluateScore(gameState, player, opponent):
    # Return infinity if a player has won in the given board
    score = checkWin(gameState)

    if score == player:
        return float("inf")
    elif score == opponent:
        return float("-inf")
    else:
        score = 0

    for i in range(0, BOARD_SIZE_Y):
        for j in range(0, BOARD_SIZE_X):
            if gameState[i][j] == 0:
                score += scoreOfCoordinate(gameState, i, j, player, opponent)

    return score

#
# Method that evaluates if a given coordinate has a possible win
# for any player. Each coordinate evaluates if a possible win can be
# found vertically, horizontally or in both diagonals.
#

def scoreOfCoordinate(gameState, i, j, player, opponent):
    score = 0

    # Check vertical line
    score += scoreOfLine(
                     gameState=gameState,
                     i=i,
                     j=j,
                     rowIncrement=-1,
                     columnIncrement=0,
                     firstRowCondition=-1,
                     secondRowCondition=BOARD_SIZE_Y,
                     firstColumnCondition=None,
                     secondColumnCondition=None,
                     player=player,
                     opponent=opponent
                 )

    # Check horizontal line
    score += scoreOfLine(
                     gameState=gameState,
                     i=i,
                     j=j,
                     rowIncrement=0,
                     columnIncrement=-1,
                     firstRowCondition=None,
                     secondRowCondition=None,
                     firstColumnCondition=-1,
                     secondColumnCondition=BOARD_SIZE_X,
                     player=player,
                     opponent=opponent
                 )

    # Check diagonal /
    score += scoreOfLine(
                     gameState=gameState,
                     i=i,
                     j=j,
                     rowIncrement=-1,
                     columnIncrement=1,
                     firstRowCondition=-1,
                     secondRowCondition=BOARD_SIZE_Y,
                     firstColumnCondition=BOARD_SIZE_X,
                     secondColumnCondition=-1,
                     player=player,
                     opponent=opponent
                 )

    # Check diagonal \
    score += scoreOfLine(
                     gameState=gameState,
                     i=i,
                     j=j,
                     rowIncrement=-1,
                     columnIncrement=-1,
                     firstRowCondition=-1,
                     secondRowCondition=BOARD_SIZE_Y,
                     firstColumnCondition=-1,
                     secondColumnCondition=BOARD_SIZE_X,
                     player=player,
                     opponent=opponent
                 )

    return score

#
# Method that searches through a line (vertical, horizontal or
# diagonal) to get the heuristic value of the given coordinate.
#

def scoreOfLine(
    gameState,
    i,
    j,
    rowIncrement,
    columnIncrement,
    firstRowCondition,
    secondRowCondition,
    firstColumnCondition,
    secondColumnCondition,
    player,
    opponent
):
    score = 0
    currentInLine = 0
    valsInARow = 0
    valsInARowPrev = 0

    # Iterate in one side of the line until a move from another
    # player or an empty space is found
    row = i + rowIncrement
    column = j + columnIncrement
    firstLoop = True
    while (
        row != firstRowCondition and
        column != firstColumnCondition and
        gameState[row][column] != 0
    ):
        if firstLoop:
            currentInLine = gameState[row][column]
            firstLoop = False
        if currentInLine == gameState[row][column]:
            valsInARow += 1
        else:
            break
        row += rowIncrement
        column += columnIncrement

    # Iterate on second side of the line
    row = i - rowIncrement
    column = j - columnIncrement
    firstLoop = True
    while (
        row != secondRowCondition and
        column != secondColumnCondition and
        gameState[row][column] != 0
    ):
        if firstLoop:
            firstLoop = False

            # Verify if previous side of line guaranteed a win on the
            # coordinate, and if not, continue counting to see if the
            # given coordinate can complete a line from in between.
            if currentInLine != gameState[row][column]:
                if valsInARow == 3 and currentInLine == player:
                    score += 1
                elif valsInARow == 3 and currentInLine == opponent:
                    score -= 1
            else:
                valsInARowPrev = valsInARow

            valsInARow = 0
            currentInLine = gameState[row][column]

        if currentInLine == gameState[row][column]:
            valsInARow += 1
        else:
            break
        row -= rowIncrement
        column -= columnIncrement

    if valsInARow + valsInARowPrev >= 3 and currentInLine == player:
        score += 1
    elif valsInARow + valsInARowPrev >= 3 and currentInLine == opponent:
        score -= 1

    return score

#
# Method that executes the first call of the minimax method and
# returns the move to be executed by the computer. It also verifies
# if any immediate wins or loses are present.
#

def bestMove(gameState, player, opponent):
    for i in range(0, BOARD_SIZE_X):
        # If moves cannot be made on column, skip it
        if gameState[0][i] != 0:
            continue

        currentMove = [0, i]

        for j in range(0, BOARD_SIZE_Y - 1):
            if gameState[j + 1][i] != 0:
                gameState[j][i] = player
                currentMove[0] = j
                break
            elif j == BOARD_SIZE_Y - 2:
                gameState[j+1][i] = player
                currentMove[0] = j+1

        winner = checkWin(gameState)
        gameState[currentMove[0]][currentMove[1]] = 0

        if winner == COMPUTER_PLAYER:
            return currentMove[1]

    for i in range(0, BOARD_SIZE_X):
        # If moves cannot be made on column, skip it
        if gameState[0][i] != 0:
            continue

        currentMove = [0, i]

        for j in range(0, BOARD_SIZE_Y - 1):
            if gameState[j + 1][i] != 0:
                gameState[j][i] = opponent
                currentMove[0] = j
                break
            elif j == BOARD_SIZE_Y - 2:
                gameState[j+1][i] = opponent
                currentMove[0] = j+1

        winner = checkWin(gameState)
        gameState[currentMove[0]][currentMove[1]] = 0

        if winner == HUMAN_PLAYER:
            return currentMove[1]

    move, score = minimax(gameState, SEARCH_DEPTH, player, opponent)
    return move[1]

#
# Method that verifies if the current board is in a winning state
# for any player, returning infinity if that is the case.
#

def checkWin(gameState):
    current = 0
    currentCount = 0
    computer_wins = 0
    opponent_wins = 0

    # Check horizontal wins
    for i in range(0, BOARD_SIZE_Y):
        for j in range(0, BOARD_SIZE_X):
            if currentCount == 0:
                if gameState[i][j] != 0:
                    current = gameState[i][j]
                    currentCount += 1
            elif currentCount == 4:
                if current == COMPUTER_PLAYER:
                    computer_wins += 1
                else:
                    opponent_wins += 1
                currentCount = 0
                break
            elif gameState[i][j] != current:
                if gameState[i][j] != 0:
                    current = gameState[i][j]
                    currentCount = 1
                else:
                    current = 0
                    currentCount = 0
            else:
                currentCount += 1

        if currentCount == 4:
            if current == COMPUTER_PLAYER:
                computer_wins += 1
            else:
                opponent_wins += 1
        current = 0
        currentCount = 0

    # Check vertical wins
    for j in range(0, BOARD_SIZE_X):
        for i in range(0, BOARD_SIZE_Y):
            if currentCount == 0:
                if gameState[i][j] != 0:
                    current = gameState[i][j]
                    currentCount += 1
            elif currentCount == 4:
                if current == COMPUTER_PLAYER:
                    computer_wins += 1
                else:
                    opponent_wins += 1
                currentCount = 0
                break
            elif gameState[i][j] != current:
                if gameState[i][j] != 0:
                    current = gameState[i][j]
                    currentCount = 1
                else:
                    current = 0
                    currentCount = 0
            else:
                currentCount += 1

        if currentCount == 4:
            if current == COMPUTER_PLAYER:
                computer_wins += 1
            else:
                opponent_wins += 1
        current = 0
        currentCount = 0

    # Check diagonal wins
    np_matrix = numpy.array(gameState)
    diags = [np_matrix[::-1,:].diagonal(i) for i in range(-np_matrix.shape[0]+1,np_matrix.shape[1])]
    diags.extend(np_matrix.diagonal(i) for i in range(np_matrix.shape[1]-1,-np_matrix.shape[0],-1))
    diags_list = [n.tolist() for n in diags]

    for i in range(0, len(diags_list)):
        if len(diags_list[i]) >= 4:
            for j in range(0, len(diags_list[i])):
                if currentCount == 0:
                    if diags_list[i][j] != 0:
                        current = diags_list[i][j]
                        currentCount += 1
                elif currentCount == 4:
                    if current == COMPUTER_PLAYER:
                        computer_wins += 1
                    else:
                        opponent_wins += 1
                    currentCount = 0
                    break
                elif diags_list[i][j] != current:
                    if diags_list[i][j] != 0:
                        current = diags_list[i][j]
                        currentCount = 1
                    else:
                        current = 0
                        currentCount = 0
                else:
                    currentCount += 1

            if currentCount == 4:
                if current == COMPUTER_PLAYER:
                    computer_wins += 1
                else:
                    opponent_wins += 1
            current = 0
            currentCount = 0

    if opponent_wins > 0:
        return HUMAN_PLAYER
    elif computer_wins > 0:
        return COMPUTER_PLAYER
    else:
        return 0

#
# Function that prints the game board, representing the player
# as a O and the computer as an X
#

def printBoard(gameState):
    for i in range(1, BOARD_SIZE_X + 1):
        sys.stdout.write(" %d " % i)

    print("")
    print("_" * (BOARD_SIZE_X * 3))
    for i in range(0, BOARD_SIZE_Y):
        for j in range(0, BOARD_SIZE_X):

            if gameState[i][j] == 1:
                sys.stdout.write("|X|")
            elif gameState[i][j] == -1:
                sys.stdout.write("|O|")
            else:
                sys.stdout.write("|-|")

        print("")

    print("_" * (BOARD_SIZE_X * 3))
    print("")

#
# Method that provides the main flow of the game, prompting the user
# to make moves, and then allowing the computer to execute a move.
# After each turn, the method checks if the board is full or if a player
# has won.
#

def playGame():
    gameState = [[0 for col in range(BOARD_SIZE_X)] for row in range(BOARD_SIZE_Y)]
    moveHeights = [0] * BOARD_SIZE_X
    player = COMPUTER_PLAYER
    opponent = HUMAN_PLAYER
    winner = 0
    gameOver = False
    remainingColumns = BOARD_SIZE_X
    print("=========================")
    print("= WELCOME TO CONNECT 4! =")
    print("=========================\n")
    printBoard(gameState)

    while True:

        while True:
            try:
                move = int(raw_input("What is your move? (Choose from 1 to %d)" % BOARD_SIZE_X))
            except ValueError:
                print("That wasn't a number! Try again.")
                continue
            if move < 1 or move > BOARD_SIZE_X:
                print("That is not a valid move. Try again.")
            elif moveHeights[move - 1] == BOARD_SIZE_Y:
                print("The chosen column is already full. Try again.")
            else:
                break

        moveHeights[move - 1] += 1
        gameState[BOARD_SIZE_Y - moveHeights[move - 1]][move - 1] = opponent
        printBoard(gameState)

        if moveHeights[move - 1] == BOARD_SIZE_Y:
            remainingColumns -= 1
        if remainingColumns == 0:
            gameOver = True
        if gameOver:
            break

        score = checkWin(gameState)
        if score == player:
            winner = player
            break
        elif score == opponent:
            winner = opponent
            break
        else:
            score = 0

        print("Now it's the computer's turn!")
        move = bestMove(gameState, player, opponent)
        if move == None:
            break

        moveHeights[move] += 1
        gameState[BOARD_SIZE_Y - moveHeights[move]][move] = player
        printBoard(gameState)

        if moveHeights[move] == BOARD_SIZE_Y:
            remainingColumns -= 1
        if remainingColumns == 0:
            gameOver = True
        if gameOver:
            break

        score = checkWin(gameState)
        if score == player:
            winner = player
            break
        elif score == opponent:
            winner = opponent
            break
        else:
            score = 0

    return winner

#
# Main execution of the game. Plays the game until the user
# wishes to stop.
#

if __name__ == "__main__":
    playing = True
    while playing:
        winner = playGame()
        if winner == COMPUTER_PLAYER:
            print("Damn! You lost!")
        elif winner == HUMAN_PLAYER:
            print("Congratulations! You won!")
        else:
            print("The board is full. This is a draw!")

        while True:
            try:
                option = raw_input("Do you want to play again? (Y/N)")
            except ValueError:
                print("Please input a correct value. Try again.")
                continue
            if option == 'Y' or option == 'y':
                break
            elif option == 'N' or option == 'n':
                playing = False
                break
            else:
                print("Please enter Y or N.")
