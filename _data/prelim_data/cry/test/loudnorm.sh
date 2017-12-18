#!/bin/bash

for i in *.mp3
do
    echo "processing $i"
    ffmpeg1 -i $i -filter:a loudnorm -ar:a 22050 "./normed/$(basename "$i" .mp3)_normed.mp3"

    echo "processed"
done

