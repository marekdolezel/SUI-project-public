#!/bin/bash

for file in $(ls ${1}); do
  python3 ./scripts/count_gameStates.py ${1}/${file} >> count.txt
done
echo "Total GameStates:"
paste -sd+ count.txt | bc