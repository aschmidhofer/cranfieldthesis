import argparse
import os

def handleDirectory(directory, prefix):
    files = next(os.walk(directory))[2]
    for fn in files:
        path = os.path.join(directory,fn)
        outpath = os.path.join(directory,prefix+fn)
        os.rename(path,outpath)
        
def processRecursive(directory, prefix):
    subdirs = next(os.walk(directory))[1]
    for subdir in subdirs:
        workdir = os.path.join(directory,subdir)
        print(workdir)
        handleDirectory(workdir, prefix)
        processRecursive(workdir, prefix)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'directory',
      type=str,
      help='Path to image directory.'
    )
    parser.add_argument(
      'prefix',
      type=str,
      help='string to add to filename.'
    )
    args = parser.parse_args()
    processRecursive(args.directory,args.prefix)



if __name__ == "__main__":
    main()
