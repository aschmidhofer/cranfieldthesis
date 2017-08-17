## this script helps with the Fd1200 dataset to remove images from the testing trainingset to create the validation set
# since the dataset is so small there cannot be two different sets 
# the validation set is thus a subset of the evaluation (testing) set

import argparse
import os
import shutil

def process(directory):
    subdirs = next(os.walk(directory))[1]
    print (subdirs)
    for sd in subdirs:
        nd = os.path.join(directory, sd)
        handleDirectory(nd)


def handleDirectory(directory):
    filenames = next(os.walk(directory))[2]
    existing = {}
    for filename in filenames:
        picid = filename.split('_')[1]
        #print(picid)
        if picid in existing:
          #remove pic
          print("removing %s" % filename)
          fullpath = os.path.join(directory,filename)
          os.remove(fullpath)
        else:
          existing[picid] = filename
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'directory',
      type=str,
      help='Path to images.'
    )
    args = parser.parse_args()
    process(args.directory)





if __name__ == "__main__":
    main()
