#!/bin/bash

for i in *.mp3
do
    echo "processing $i"
    if [ $(soxi -c $i) -eq 2 ]
    then
        sox $i "/home/aeatda/_windev/projects/proj3/_data/mp3s_processed/giggling/$(basename "$i" .mp3)_mono.mp3" remix 1,2
    else
        cp $i "/home/aeatda/_windev/projects/proj3/_data/mp3s_processed/giggling/$(basename "$i" .mp3)_mono.mp3"

    fi
    echo "processed"
done
