#!/usr/bin/env python3


from matplotlib import pyplot as plt
import numpy as np
import sys
import pickle
from scripts.gameSerialize import *
bSize = 34

def reconstruct_Mod_from_vector(gameStateVector):
    M = np.zeros((bSize, bSize))
    o = np.zeros((bSize))
    d = np.zeros((bSize))

    gameStateVectorIdx = 0
    for x in range(0, bSize):
        for y in range(x, bSize):
            M[x, y] = gameStateVector[gameStateVectorIdx]
            gameStateVectorIdx = gameStateVectorIdx + 1
    for x in range(0, bSize):
        o[x] = gameStateVector[gameStateVectorIdx]
        gameStateVectorIdx = gameStateVectorIdx + 1

    for x in range(0, bSize):
        d[x] = gameStateVector[gameStateVectorIdx]
        gameStateVectorIdx = gameStateVectorIdx + 1
    return M, o, d


def helper_plot_matrix_M(M):
    range = np.arange(1, 35, 1)
    plt.imshow(M, interpolation='nearest')
    plt.xticks(np.arange(0, 35, 2));
    plt.yticks(np.arange(0, 35, 2));
    plt.show()


if __name__ == '__main__':
    with open(sys.argv[1], "rb") as file:
        tuplePlayersArrOfBoard = pickle.load(file)

    players, arrOfBoards = tuplePlayersArrOfBoard

    for board in arrOfBoards:
        gameStateVector = serialize_game_state(board)
        M,o,d = reconstruct_Mod_from_vector(gameStateVector)
        # helper_plot_matrix_M(M)

        print("o",o)
        print("d",d,"\n\n")

        print("Visualization done")
