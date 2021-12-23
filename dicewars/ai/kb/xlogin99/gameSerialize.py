from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os
def helper_plot_matrix_M(M):
    range = np.arange(1, 35, 1)
    plt.imshow(M, interpolation='nearest')
    plt.show()


def serialize_game_state(board):
    bSize = 34
    adjacencyMatrix = np.zeros((bSize, bSize))
    ownerShipVector = np.zeros((bSize))
    diceVector = np.zeros((bSize))

    for i in range(1, bSize + 1):
        area = board.get_area(i)
        area0BasedIndex = area.name - 1

        ownerShipVector[area0BasedIndex] = area.owner_name
        diceVector[area0BasedIndex] = area.dice

        for neighbour1BasedIndex in area.neighbours:
            neighbour0BasedIndex = neighbour1BasedIndex - 1

            adjacencyMatrix[area0BasedIndex, neighbour0BasedIndex] = 1
    return adjacencyMatrix, ownerShipVector, diceVector

def game_state_to_vector( M, o, d):
    bSize = 34
    gameStateVectorSize = (bSize ** 2 - bSize) / 2 + bSize + 2 * bSize
    gameStateVector = np.zeros((int(gameStateVectorSize)))
    gameStateVectorIdx = 0
    for x in range(0, bSize):
        for y in range(x, bSize):
            gameStateVector[gameStateVectorIdx] = M[x, y]
            gameStateVectorIdx = gameStateVectorIdx + 1

    for x in range(0, bSize):
        gameStateVector[gameStateVectorIdx] = o[x]
        gameStateVectorIdx = gameStateVectorIdx + 1

    for x in range(0, bSize):
        gameStateVector[gameStateVectorIdx] = d[x]
        gameStateVectorIdx = gameStateVectorIdx + 1

    return gameStateVector

def save_game_state_vector_to_file(gameStateVector):
    if not os.path.exists("gameStates"):
        os.mkdir("gameStates")
        
    now = datetime.now()
    fileName = now.strftime("%m%d_%Y_%H%M%S")
    np.save("gameStates/" + fileName, gameStateVector)