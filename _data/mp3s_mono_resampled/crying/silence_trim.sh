#!/bin/bash

for i in *.mp3
do
    echo "processing $i"
    
    tempadd="/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/crying/proc1/$(basename "$i" .mp3).mp3"
    tempadd2="/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/crying/proc2/$(basename "$i" .mp3).mp3"

    sox $i $tempadd silence 1 0.3 1% 
    dur=$(soxi -D $tempadd)
    echo $dur

    if [ ${dur%%.*} -gt 4 ]
    then
        sox $tempadd $tempadd2 trim 0 3.92
    else
        echo "hi"
#        cp $i "/home/aeatda/_windev/projects/proj3/_data/mp3s_mono_resampled/crying/proc2/$(basename "$i" .mp3).mp3"

    fi
    echo "processed"
done
