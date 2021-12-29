#!/usr/bin/env python3
import pickle
import os,sys
import numpy as np
from scripts.gameSerialize import serialize_game_state, save_game_state_vector_to_file
# Pickle file format:
# containst tuple of objects (players, arrOfBoards),
#   where 'players' object is a dictionary mapping player_name to player_nickname
#       new mapping that is same for all games must be created!
#   and 'arrOfBoards' is a board object representing the state of the game

def compute_globalPlayersDict(players, globalPlayersDict, maxIndex):
    if (len(globalPlayersDict.values()) == 0):
        globalPlayersDict = {value:key for key, value in players.items()}
        maxIndex = len(globalPlayersDict)
    else:
        for value in players.values():
            if value not in globalPlayersDict.keys():
                globalPlayersDict[value] = str(maxIndex+1)
                maxIndex = maxIndex + 1
    return globalPlayersDict, maxIndex

def compute_winner(lastBoard):
    winner = -1
    for i in range(1, 34 + 1):
        area = lastBoard.get_area_by_name(i)
        if winner == -1:
            winner = area.owner_name
        elif area.owner_name == winner:
            pass
        else:
            return -1
    return winner

if __name__ == '__main__':
    globalPlayersDict = {}
    maxIndex = 0
    gameStateVectorsArray = np.zeros(664)
    for file in os.listdir(sys.argv[1]):
        with open(sys.argv[1]+"/"+file, "rb") as handle:
            tuplePlayersArrOfBoard = pickle.load(handle)

            players, arrOfBoards = tuplePlayersArrOfBoard
            globalPlayersDict, maxIndex = compute_globalPlayersDict(players, globalPlayersDict, maxIndex)

            winner = compute_winner(arrOfBoards[-1])
            if (winner == -1):
                print("Skipping the file",{file})
                break
            winner = globalPlayersDict[players[winner]] # Translate winner to globalIndex

            # gameStateVectorsArray = serialize_game_state(arrOfBoards[0], players, globalPlayersDict)
            # gameStateVectorsArray = np.append(gameStateVectorsArray, winner) # add winner to the vector
            for board in arrOfBoards[1:]:
                gameStateVector = serialize_game_state(board, players, globalPlayersDict)
                gameStateVector = np.append(gameStateVector, winner)  # add winner to the vector
                gameStateVectorsArray = np.vstack((gameStateVectorsArray, gameStateVector))
            gameStateVectorsArray = np.delete(gameStateVectorsArray, (0), axis=0)
    save_game_state_vector_to_file(gameStateVectorsArray)

