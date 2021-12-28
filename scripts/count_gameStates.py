#!/usr/bin/env python3


import sys
import pickle




if __name__ == '__main__':
    with open(sys.argv[1], "rb") as file:
        tuplePlayersArrOfBoard = pickle.load(file)

    players, arrOfBoards = tuplePlayersArrOfBoard

    print(len(arrOfBoards))
