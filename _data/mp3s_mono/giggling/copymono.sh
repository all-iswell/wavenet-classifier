#!/bin/bash

for i in *.mp3
do
    echo "processing $i"
    sox $i "lim_$i" compand .05,.3 9:-15,0,-8

    if [ $(soxi -r $i) -ne 22050 ]
    then
        sox "lim_$i" -r 22050 "/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/giggling/$(basename "$i" .mp3)_mono.mp3"
    else
        cp "lim_$i" "/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/giggling/$(basename "$i" .mp3)_mono.mp3"

    fi
    echo "processed"
done
