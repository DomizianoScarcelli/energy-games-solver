#!/bin/bash

STRATEGY="incremental_bellman_ford" #none, bellman_ford, incremental_bellman_ford
PROBABILITIES=(0.1 0.2 0.3 0.4 0.5)
NUM_NODES=(100 200 300 400 500 1000 2000 5000)
REPEATS=3
SAVE_ARENA=false

for i in $(seq 1 $REPEATS)
do
    echo "Repeat number $i"
    for p in "${PROBABILITIES[@]}"
    do
        for n in "${NUM_NODES[@]}"
        do
            echo "Generating arena with $n nodes, edge probability $p and strategy $STRATEGY"
            if [ "$SAVE_ARENA" = true ]; then
                python -m run --generate --num-nodes=$n --edge-probability=$p --strategy="$STRATEGY" --save-arena
            else 
                python -m run --generate --num-nodes=$n --edge-probability=$p --strategy="$STRATEGY"
            fi
        done
    done
done

echo "Done"
