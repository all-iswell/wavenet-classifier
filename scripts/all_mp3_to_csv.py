#! /home/aeatda/anaconda3/envs/proj3/bin/python

import sys
import os
import re
import pydub as pdb
import csv
# from freesound_api import *


# This script must be located in project/scripts, with the data in
# project/_data/prelim_data, to work properly.
# Otherwise the paths must be modified.

def write_to_csv(category):
    cat_path = '../_data/prelim_data/{}/'.format(category)
    
    names = [name for name in os.listdir(cat_path)
                   if re.search(r'.mp3$', name)]

    try:
        with open('{0}csv/{1}.csv'.format(cat_path, category), 'w') as outfile:
            writer = csv.writer(outfile)

            for name in names:
                file_arr = pdb.AudioSegment.from_mp3(cat_path + name)\
                              .get_array_of_samples().tolist()

                print("Writing file {}".format(name))
                writer.writerow(file_arr)

    except Exception as e:
        print("Failed with error: {}".format(e))


if __name__ == "__main__":
    args = sys.argv[1:]
    write_to_csv(*args)
    print("Done")
