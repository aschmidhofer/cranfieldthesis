import PIL
from PIL import Image
import argparse
import os
import numpy as np

REQUIRED_WIDTH = 299
REQUIRED_HEIGHT = 299
SCALE = True

def convertToHSV(img, keep_aspect=False):
    hsvimg = img.convert('HSV')
    #dd = hsvimg.getdata(2) # band 2 is V
    pixels = hsvimg.width * hsvimg.height
    #newimg = Image.new('RGB', hsvimg.size)
    #newimg.putdata(dd)
    #return newimg
    #return hsvimg.convert('RGB')
    
    data = np.array(hsvimg)
    h, s, v = data.T
    data[:,:,2] = 255
    
    newimg = Image.fromarray(data,'HSV')
    return newimg.convert('RGB')
     

def handleDirectory(directory, outputdir):
    files = next(os.walk(directory))[2]
    for fn in files:
        path = os.path.join(directory,fn)
        if fn.endswith(".jpg"):
            img = Image.open(path)
            img = convertToHSV(img, SCALE)
            prefix = "value_"
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



#path = '/home/andy/imgs/Food-11-resized/training/2/resized_2_3.jpg'
#img = Image.open(path)
#img = convertToHSV(img)
#img.save("hallo.jpg")



if __name__ == "__main__":
    main()
