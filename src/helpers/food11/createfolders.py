## this script helps put the dataset images in the right folders
# the images are stored in one directory with the class name before the underscore
# they will be put in directories that correspond to the class name

import argparse
import os
import shutil

def processDataset(directory, outdir):
    filenames = next(os.walk(directory))[2]
    copy = True
    if outdir is None:
        copy = False
        outdir = directory
    for filename in filenames:
        cat = filename.split('_')[0]
        path = outdir + '/' + cat
        if not os.path.exists(path):
            os.mkdir(path)
        src = directory+'/'+filename
        dst = path+'/'+filename
        if copy:
            shutil.copyfile(src, dst)
        else:
            shutil.move(src, dst)
        

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'directory',
      type=str,
      help='Path to images.'
    )
    parser.add_argument(
      '-output_directory',
      type=str,
      default=None,
      help='Path to store cropped images.'
    )
    args = parser.parse_args()
    processDataset(args.directory,args.output_directory)





if __name__ == "__main__":
    main()
