## this script helps for the Food-101 dataset
# the images are already in folders according to labels
# they just need to be split for test and training set

import argparse
import os
import shutil

def handlefilenamefile(imgdir,fn,outputdir):
    print("start")
    files = [line.rstrip('\n')+".jpg" for line in open(fn)]
    
    for subpath in files:
        src = os.path.join(imgdir, subpath)
        dst = os.path.join(outputdir, subpath)
        dirct = os.path.dirname(dst)
        if not os.path.exists(dirct):
            os.makedirs(dirct)
        shutil.copyfile(src, dst)
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--directory',
      type=str,
      default='.',
      help='Path to image directory.'
    )
    parser.add_argument(
      'file',
      type=str,
      help='path to file containing the filenames to put in output folder.'
    )
    parser.add_argument(
      '--output',
      type=str,
      default='./output',
      help='Path to output directory.'
    )
    args = parser.parse_args()
    handlefilenamefile(args.directory,args.file,args.output)



if __name__ == "__main__":
    main()
