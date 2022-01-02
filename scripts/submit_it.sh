#!/bin/bash

SUBMIT_DIR=submission
LEADER=xdolez67
SUPP_DIR=$SUBMIT_DIR/supp-$LEADER

mkdir $SUBMIT_DIR
mkdir -p $SUPP_DIR
cp -r  dicewars/ai/kb/xdolez67 $SUBMIT_DIR

cp ./scripts/process_pickle_to_np_array.py $SUPP_DIR
cp ./scripts/gameSerialize.py $SUPP_DIR
cp ./scripts/network.py $SUPP_DIR
cp ./scripts/extract_subset_fromH5.py $SUPP_DIR

cd $SUBMIT_DIR
zip -r $LEADER *
cp  $LEADER.zip ../
