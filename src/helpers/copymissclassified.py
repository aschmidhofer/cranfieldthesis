## this script helps to copy misclassified images to another folder
# which allows a better overfiew on the images
# as an imput it takes the printed output of the retrain.py 

import argparse
import os
import shutil

def process(misfile, outdir):
    with open(misfile) as f:
      lines = f.readlines()
      for line in lines:
        parts = line.split()
        if(len(parts)!=2):
          print(parts)
          print("..skipped")
          continue
        srcpath = parts[0]
        catgr = parts[1]
        folder = os.path.join(outdir,catgr)
        if not os.path.exists(folder):
          os.makedirs(folder)
        dst = os.path.join(folder,os.path.basename(srcpath))
        shutil.copyfile(srcpath, dst)
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'file',
      type=str,
      help='Path to the file containing the misprint paths.'
    )
    parser.add_argument(
      '-output_directory',
      type=str,
      default='./output',
      help='Path to store the images.'
    )
    args = parser.parse_args()
    process(args.file,args.output_directory)





if __name__ == "__main__":
    main()
