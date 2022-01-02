#!/usr/bin/env python3
import pickle
from multiprocessing import Process, cpu_count
import os,sys
import numpy as np
import h5py
from scripts.gameSerialize import serialize_game_state, save_game_state_vector_to_file, serialize_game_state_fast
# Pickle file format:
# containst tuple of objects (players, arrOfBoards),
#   where 'players' object is a dictionary mapping player_name to player_nickname
#       new mapping that is same for all games must be created!
#   and 'arrOfBoards' is a board object representing the state of the game
# globalPlayersDict = {'kb.sdc_pre_at (AI)': 2, 'kb.stei_dt (AI)': 3, 'kb.xlogin99 (AI)': 4, 'kb.stei_at (AI)': 1}


def compute_globalPlayersDict(players, globalPlayersDict, maxIndex):
    if (len(globalPlayersDict.values()) == 0):
        globalPlayersDict = {value:key for key, value in players.items()}
        maxIndex = len(globalPlayersDict)
    else:
        for value in players.values():
            if value not in globalPlayersDict.keys():
                globalPlayersDict[value] = str(maxIndex+1)
                maxIndex = maxIndex + 1
                print("Adding to globalPlayersDict: winnerLoc_name {}, winner_nickname {}, winnerGlob_name {}".format(int([k for k,v in players.items() if v == value][0]), value, globalPlayersDict[value]))
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
        assert (type(winner) == int)
    return winner


def process_pickles(files_to_process, rootdir):
    globalPlayersDict = {}
    maxIndex = 0
    winnerStats = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0}
    for file in files_to_process:


        with open(rootdir+"/"+file, "rb") as handle:
            tuplePlayersArrOfBoard = pickle.load(handle)

            players, arrOfBoards = tuplePlayersArrOfBoard
            globalPlayersDict, maxIndex = compute_globalPlayersDict(players, globalPlayersDict, maxIndex)

            winner = compute_winner(arrOfBoards[-1])
            if (winner == -1):
                print("Skipping the file",{file})
                break
            winner = globalPlayersDict[players[winner]] # Translate winner to globalIndex
            winnerStats[str(winner)]  = winnerStats[str(winner)] + 1
    print(winnerStats)
    print(globalPlayersDict)





if __name__ == '__main__':
    # Post process files in parallel
    ListOfFilenames = os.listdir(sys.argv[1])
    process_pickles(ListOfFilenames, sys.argv[1])
