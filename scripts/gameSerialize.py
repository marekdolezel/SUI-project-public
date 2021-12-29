from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
import os
def helper_plot_matrix_M(M):
    range = np.arange(1, 35, 1)
    plt.imshow(M, interpolation='nearest')
    plt.show()

def serialize_game_state(board, players, globalPlayersDict):
    M,o,d = _serialize_game_state(board, players, globalPlayersDict)
    return _game_state_to_vector(M,o,d)

def _serialize_game_state(board, players, globalPlayersDict):
    bSize = 34
    adjacencyMatrix = np.zeros((bSize, bSize))
    ownerShipVector = np.zeros((bSize))
    diceVector = np.zeros((bSize))

    # board.get_board()

    for i in range(1, bSize + 1):
        area = board.get_area_by_name(i) # Areas are indexed from 1 to 34
        owner_name = area.owner_name
        areaIndex = area.name - 1

        new_owner_name = globalPlayersDict[players[owner_name]]
        ownerShipVector[areaIndex] = new_owner_name
        diceVector[areaIndex] = area.dice

        neighbours = area.get_adjacent_areas()
        for neighbour_area in neighbours:
            neighbourIndex = neighbour_area.name - 1

            adjacencyMatrix[areaIndex, neighbourIndex] = 1
    return adjacencyMatrix, ownerShipVector, diceVector

def _game_state_to_vector( M, o, d):
    bSize = 34
    gameStateVectorSize = (bSize ** 2 - bSize) / 2 + bSize
    gameStateVector = np.zeros((int(gameStateVectorSize)))
    gameStateVectorIdx = 0

    # Transform upper triangle (including diagonal) to a vector (ugly code)
    for x in range(0, bSize):
        for y in range(x, bSize):
            gameStateVector[gameStateVectorIdx] = M[x, y]
            gameStateVectorIdx = gameStateVectorIdx + 1

    gameStateVector = np.append(gameStateVector, o)
    gameStateVector = np.append(gameStateVector, d)

    assert ((bSize ** 2 - bSize) / 2 + bSize + 2 * bSize) == len(gameStateVector)

    return gameStateVector

def save_game_state_vector_to_file(gameStateVector):
    if not os.path.exists("gameStates"):
        os.mkdir("gameStates")
        
    now = datetime.now()
    fileName = now.strftime("%m%d_%Y_%H%M%S")
    np.save("gameStates/" + fileName, gameStateVector)