
from PIL import Image
import argparse
import os



def cropImage(directory, imgid, x1, y1, x2, y2, outputdir):
    path = directory + '/' + imgid + '.jpg'
    try:
        img = Image.open(path)
        crop = img.crop((x1, y1, x2, y2))
        newName = outputdir + '/' + imgid + '.jpg'
        crop.save(newName)
        #print(newName)
    except:
        print("ERROR: "+path)

def handleDirectory(directory, outputdir):
    bbfilename = directory + '/' + "bb_info.txt"
    with open(bbfilename, 'r') as infofile:
        infofile.readline() #ignore header
        for line in infofile:
            #print(line.split())
            info = line.split()
            cropImage(directory, info[0], int(info[1]), int(info[2]), int(info[3]), int(info[4]), outputdir)
           
           
def processDataset(directory, outputbasedir):
    subdirs = next(os.walk(directory))[1]
    for subdir in subdirs:
        print (subdir)
        outdir = outputbasedir + '/' + subdir
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        workdir = directory + '/'+ subdir
        handleDirectory(workdir, outdir)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'directory',
      type=str,
      help='Path to image database.'
    )
    parser.add_argument(
      '-output_directory',
      type=str,
      default='.',
      help='Path to store cropped images.'
    )
    args = parser.parse_args()
    processDataset(args.directory,args.output_directory)





if __name__ == "__main__":
    main()
