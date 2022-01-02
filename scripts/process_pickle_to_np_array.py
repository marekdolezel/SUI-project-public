#!/usr/bin/env python3
import pickle
from multiprocessing import Process, cpu_count
import os,sys
import numpy as np
import h5py
from scripts.gameSerialize import serialize_game_state, serialize_game_stateNoTraslation
# Pickle file format:
# containst tuple of objects (players, arrOfBoards),
#   where 'players' object is a dictionary mapping player_name to player_nickname
#       new mapping that is same for all games must be created!
#   and 'arrOfBoards' is a board object representing the state of the game
# globalPlayersDict = {'kb.sdc_pre_at (AI)': 2, 'kb.stei_dt (AI)': 3, 'kb.xlogin99 (AI)': 4, 'kb.stei_at (AI)': 1}

h5dataset = h5py.File('MyDataset.h5', 'a')

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
        print("winner is ",winner)
        assert (type(winner) == int)
    return winner


def process_pickles(files_to_process, rootdir):
    globalPlayersDict = {}
    maxIndex = 0
    gameStateVectorsArray = np.zeros(664)

    processingFirstFile = True
    for file in files_to_process:
        print("Processing file", {file})
        gameStateVectorsArray = np.zeros(664)
        with open(rootdir+"/"+file, "rb") as handle:
            tuplePlayersArrOfBoard = pickle.load(handle)

            players, arrOfBoards = tuplePlayersArrOfBoard
            # globalPlayersDict, maxIndex = compute_globalPlayersDict(players, globalPlayersDict, maxIndex)

            winner = compute_winner(arrOfBoards[-1])
            if (winner == -1):
                print("Skipping the file",{file})
                break
            # "globalIndexes" were pretty bad idea afterall :-(
            # winner = globalPlayersDict[players[winner]] # Translate winner to globalIndex

            # gameStateVectorsArray = serialize_game_state(arrOfBoards[0], players, globalPlayersDict)
            # gameStateVectorsArray = np.append(gameStateVectorsArray, winner) # add winner to the vector
            for board in arrOfBoards:
                # gameStateVector = []
                gameStateVector = serialize_game_stateNoTraslation(board)
                # gameStateVector = np.array(gameStateVector)
                # a = serialize_game_state(board, players, globalPlayersDict)
                # if not np.array_equal(a, gameStateVector):
                #     exit(2)
                gameStateVector = np.append(gameStateVector, int(winner))  # add winner to the vector
                gameStateVectorsArray = np.vstack((gameStateVectorsArray, gameStateVector))
            gameStateVectorsArray = np.delete(gameStateVectorsArray, (0), axis=0)

            # Create a file if iteration=0, otherwise append data to the file
            if processingFirstFile == True: #
                h5dataset.create_dataset('data', data=gameStateVectorsArray, compression="gzip", chunks=True, maxshape=(None, 664))
                processingFirstFile = False
            else:
                print("gameStateConfigs in file", gameStateVectorsArray.shape[0], {file})
                oldDatasetRows=h5dataset['data'].shape[0]
                h5dataset['data'].resize((h5dataset['data'].shape[0] + gameStateVectorsArray.shape[0]), axis=0)
                h5dataset['data'][oldDatasetRows:,:] = gameStateVectorsArray
            print("'data' chunk has shape:{}".format( h5dataset['data'].shape))
    h5dataset.close()

# 1 434 390
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


if __name__ == '__main__':
    # Post process files in parallel
    ListOfFilenames = os.listdir(sys.argv[1])

    # One if processing all files, other number if you want to process less files
    if int(sys.argv[3]) != 1:
        Parts = split(ListOfFilenames, int(sys.argv[3]))
        ListOfFilenames = list(Parts)[1]


    ListOfProcesses = []
    Processors = cpu_count()  # n of processors you want to use
    print("detected ",cpu_count(), "processors")
    print("will be processing:", len(ListOfFilenames)," Files...")
    # Divide the list of files in 'n of processors' Parts
    Parts = split(ListOfFilenames, Processors)
    if (sys.argv[2] == "single"): # if single is specified process files in sequential order
        process_pickles(ListOfFilenames, sys.argv[1] )
    else:
        for part in list(Parts):
            print("Starting process")
            p = Process(target=process_pickles, args=(part, sys.argv[1]))
            p.start()
            ListOfProcesses.append(p)
        for p in ListOfProcesses:
            p.join()