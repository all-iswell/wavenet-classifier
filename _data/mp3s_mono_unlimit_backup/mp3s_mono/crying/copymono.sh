#!/bin/bash

for i in *.mp3
do
    echo "processing $i"
    if [ $(soxi -r $i) -ne 22050 ]
    then
        sox $i $i compand .05,.3 9:-15,0,-8
        sox $i -r 22050 "/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/crying/$(basename "$i" .mp3)_mono.mp3"
    else
        cp $i "/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/crying/$(basename "$i" .mp3)_mono.mp3"

    fi
    echo "processed"
done
