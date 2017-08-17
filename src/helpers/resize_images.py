import PIL
from PIL import Image
import argparse
import os

REQUIRED_WIDTH = 299
REQUIRED_HEIGHT = 299
SCALE = True

def resizeImage(img, keep_aspect=False):
    if not keep_aspect:
        return img.resize((REQUIRED_WIDTH,REQUIRED_HEIGHT), PIL.Image.ANTIALIAS)
    
    # scale to the smaller side
    smaller = min(img.size[0],img.size[1])
    scalefactor = (REQUIRED_WIDTH/float(smaller)) # only works with REQUIRED_WIDTH=REQUIRED_HEIGHT
    neww = int((float(img.size[0])*float(scalefactor)))
    newh = int((float(img.size[1])*float(scalefactor)))
    img = img.resize((neww,newh), PIL.Image.ANTIALIAS)
    
    # now crop middle
    basew = int((neww - REQUIRED_WIDTH)/2)
    baseh = int((newh - REQUIRED_HEIGHT)/2)
    return img.crop((basew, baseh, basew+REQUIRED_WIDTH, baseh+REQUIRED_HEIGHT))
     


def handleDirectory(directory, outputdir):
    files = next(os.walk(directory))[2]
    for fn in files:
        path = os.path.join(directory,fn)
        if fn.endswith(".jpg"):
            img = Image.open(path)
            img = resizeImage(img, SCALE)
            prefix = "resized_"
            if SCALE:
                prefix = "scaled_"
            outpath = os.path.join(outputdir,prefix+fn)
            img.save(outpath)
        else:
            print("ignoring "+path)

def processRecursive(directory, outputbasedir):
    subdirs = next(os.walk(directory))[1]
    for subdir in subdirs:
        outdir = os.path.join(outputbasedir,subdir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        workdir = os.path.join(directory,subdir)
        print(workdir)
        handleDirectory(workdir, outdir)
        processRecursive(workdir, outdir)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'directory',
      type=str,
      help='Path to image directory.'
    )
    parser.add_argument(
      '-output_directory',
      type=str,
      default='.',
      help='Path to store resized images.'
    )
    args = parser.parse_args()
    processRecursive(args.directory,args.output_directory)





if __name__ == "__main__":
    main()
