#!/usr/bin/env python3
import os,sys
import numpy as np
import h5py

if __name__ == '__main__':
    h5dataset = h5py.File('UniformDataset210K.h5', 'a')

    winnerArrays = {}

    h5f = h5py.File(sys.argv[1], 'r')
    data = h5f['data'][:]
    h5f.close()

    np.random.shuffle(data)

    for i in range(1,7):
        winnerArrays[str(i)] = data[np.where(data[:, 663] == i)]
        np.random.shuffle(winnerArrays[str(i)])
        winnerArrays[str(i)] = winnerArrays[str(i)][0:35000,:]

    # merge np array into one:
    npArray = winnerArrays['1']
    for i in range(2, 7):
        npArray = np.vstack((npArray, winnerArrays[str(i)]))
    np.random.shuffle(npArray)

    h5dataset.create_dataset('data', data=npArray, compression="gzip", chunks=True, maxshape=(None, 664))


