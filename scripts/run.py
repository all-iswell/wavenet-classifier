#! /home/aeatda/anaconda3/envs/proj3/bin/python

import sys
from freesound_api import *

if __name__ == "__main__":
    func = sys.argv[1]
    args = sys.argv[2:]
    eval("{}(*args)".format(func))
    print("Done")
