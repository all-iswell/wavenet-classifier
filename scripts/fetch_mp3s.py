#! /home/aeatda/anaconda3/envs/proj3/bin/python

import sys
from freesound_api import *

if __name__ == "__main__":
    args = sys.argv[1:]
    fetch_mp3s(*args)
    print("Done")
