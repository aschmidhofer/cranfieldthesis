import argparse
import os
import shutil

def handlefilenamefile(imgdir,fn,outputdir,lbls):
    print("start")
    filenames = [line.rstrip('\n') for line in open(fn)]
    
    labels = None
    if lbls:
        labels = {}
        with open(lbls) as f:
            for line in f:
               (key, val) = line.split(None, 1)
               labels[key] = val.strip()

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    for filename in filenames:
        src = os.path.join(imgdir, filename)
        if filename.startswith("_"): # strip it off
            filename = filename[1:]
        
        if labels:
            #print("%s has label %s" %(filename, labels[filename]))
            label = labels[filename]
            if len(label.split())>1:
              print("skipping double label %s (%s)" %(filename, labels[filename]))
              continue
              
            classdir = os.path.join(outputdir, label)
            if not os.path.exists(classdir):
                os.mkdir(classdir)
            dst = os.path.join(classdir, filename)
            shutil.copyfile(src, dst)
            
        else:
            dst = os.path.join(outputdir, filename)
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
      help='path to file containing the filenames.'
    )
    parser.add_argument(
      '--output',
      type=str,
      default='./output',
      help='Path to output directory.'
    )
    parser.add_argument(
      '--labels',
      type=str,
      default=None,
      help='Path to labels file.'
    )
    args = parser.parse_args()
    handlefilenamefile(args.directory,args.file,args.output,args.labels)



if __name__ == "__main__":
    main()
