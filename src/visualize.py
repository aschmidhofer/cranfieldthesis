# Copyright 2017, Andreas Schmidhofer. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pickle
import argparse
import matplotlib.pyplot as plt
import fnmatch
import os
import matplotlib.gridspec as gridspec



def plotit(stats, ax):
  ax.plot(stats['steps'], stats['training'], label='training')
  ax.plot(stats['steps'], stats['validation'], label='validation')
  
def plotlr(stats, ax):
  if 'learning_rate' in stats:
    ax.plot(stats['steps'], stats['learning_rate'],'g-', label='learning_rate')
def plotce(stats, ax):
  if 'cross_entropy' in stats:
    ax.plot(stats['steps'], stats['cross_entropy'],'r-', label='cross_entropy')
  
def getdata(filepath):
  data = []
  
  with open(filepath, "rb") as picklefile:
    stats = pickle.load(picklefile)
    print(stats['evaluation'])
    #print(stats['training'])
    data = stats
  
  return data


def plotsingle(filepath, show=False):
    stats = getdata(filepath)
    #fig, ax = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[2, 1, 1]},figsize=(6,8))
    
    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(2, 3)
    #ax = [plt.subplot(gs[0:2, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1])]
    ax0 = fig.add_subplot(gs[0:2, 0:2])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
    ax = [ax0,ax1,ax2]
    
    plotit(stats, ax[0])
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[2].set_xlabel('epoch')
    ax[1].yaxis.tick_right()
    ax[2].yaxis.tick_right()
    ax[0].legend(loc='lower right')
    plotlr(stats, ax[1])
    ax[1].set_ylabel('learning rate')
    plotce(stats, ax[2])
    ax[2].set_ylabel('cross entropy')
    ax[0].set_title('evaluation accuracy = %.1f%%'%(stats['evaluation']*100))
    
    outfile = filepath + '_landscape.png'
    plt.savefig(outfile) #save to file
    if show:
      plt.show()
    plt.close(fig)



def plotdouble(filepath, overlayfile, show=False):
    stats1 = getdata(filepath)
    stats2 = None
    if not overlayfile=='None':
      stats2 = getdata(overlayfile)
    
    fig = plt.figure()
    ax = plt.gca()
    
    stats1['training'] = [i * 100 for i in stats1['training']]
    stats1['validation'] = [i * 100 for i in stats1['validation']]
    
    plotit(stats1, ax)
    if stats2:
      stats2['training'] = [i * 100 for i in stats2['training']]
      stats2['validation'] = [i * 100 for i in stats2['validation']]
      plotit(stats2, ax)
    ax.set_ylabel('accuracy in %')
    ax.set_xlabel('epoch')
    ax.legend(loc='lower right')
    #ax.set_title('evaluation accuracy = %.1f%%'%(stats['evaluation']*100))
    ax.set_title('An example of overfitting')
    
    outfile = filepath + '_'+ overlayfile+'_multi.pgf'#png'
    plt.savefig(outfile) #save to file
    if show:
      plt.show()
    #plt.close(fig)
    
    

def savemultiple(directory):
  for filename in os.listdir(directory):
    if fnmatch.fnmatch(filename, '*.data'):
      plotsingle(os.path.join(directory, filename))
  
def plotmultiple(directory):
  filesg = []
  filesam = []
  filesag = []
  for filename in os.listdir(directory):
    if fnmatch.fnmatch(filename, 'grad*_dynamic.data'):
        print (filename)
        filesg.append(filename)
    if fnmatch.fnmatch(filename, 'adam*_dynamic.data'):
        filesam.append(filename)
    if fnmatch.fnmatch(filename, 'adag*_dynamic.data'):
        filesag.append(filename)
  
  filesg.sort(key=len, reverse=True)
  filesam.sort(key=len, reverse=True)
  filesag.sort(key=len, reverse=True)
  cnt = len(filesg)
  assert(cnt == len(filesag))
  assert(cnt == len(filesam))
  f, axarr = plt.subplots(cnt, 3, sharex=True, sharey=True)
  for i in range(cnt):
    plotfile(os.path.join(directory,filesg[i]), axarr[i, 0])
    plotfile(os.path.join(directory,filesam[i]), axarr[i, 1])
    plotfile(os.path.join(directory,filesag[i]), axarr[i, 2])
  
  axarr[0, 0].set_title('Gradient Decend')
  axarr[0, 1].set_title('Adam')
  axarr[0, 2].set_title('Adagrad')
  axarr[0, 0].set_ylabel('learning rate = 0.001')
  axarr[1, 0].set_ylabel('learning rate = 0.01')
  axarr[2, 0].set_ylabel('learning rate = 0.1')
  axarr[3, 0].set_ylabel('learning rate = 1')
    
  #plt.ylabel('accuracy')
  #plt.xlabel('training step')
  #plt.legend()
  plt.show()

def plotlrcomp(args):
  lrs = ["0p1", "0p01", "1p0"]#, "0p16", "100p0"]
  lrlabels = ["normal", "low", "high"]#, "ideal", "huge"]
  
  
  fig = plt.figure()
  ax = plt.gca()
  
  #ax.set_title('evaluation accuracy = %.1f%%'%(stats['evaluation']*100))
  
      
  for i, lr in enumerate(lrs):
  
    filepath = "./lrcomp/gradientdescent_vS_%s_200.data" %lr
    stats = getdata(filepath)
    stats['training'] = [i * 100 for i in stats['training']]
    stats['validation'] = [i * 100 for i in stats['validation']]
    
    
    
    if i==2:
      ax.plot(stats['steps'], stats['validation'], label='%s'%lrlabels[i], color='red')
    else:
      ax.plot(stats['steps'], stats['validation'], label='%s'%lrlabels[i])
    
  #   ax.set_ylim([0,100])
  ax.set_ylabel('accuracy in %')
  ax.set_xlabel('epoch')
  ax.legend(loc='lower right')
  ax.set_title('A comparison of learning rates')
  
  outfile = "lrcomp.pgf"#"lrcomp.png"
  plt.savefig(outfile) #save to file
  plt.show()
  
    
def plotalrgd(args):
    filepath = args.file
  
    stats = getdata(filepath)
    stats['training'] = [i * 100 for i in stats['training']]
    stats['validation'] = [i * 100 for i in stats['validation']]
    if args.hidelr:
      fig, ax = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[2, 1]},figsize=(6,6))
      crax = ax[1]
    else:
      fig, ax = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[2, 1, 1]},figsize=(6,8))
      crax = ax[2]
    plotit(stats, ax[0])
    ax[0].set_ylabel('accuracy in %')
    crax.set_xlabel('epoch')
    ax[0].legend(loc='lower right')
    if not args.hidelr:
      plotlr(stats, ax[1])
      ax[1].set_ylabel('learning rate')
    plotce(stats, crax)
    crax.set_ylabel('cross entropy')
    #ax[0].set_title('evaluation accuracy = %.1f%%'%(stats['evaluation']*100))
    #ax[0].set_title('Adaptive learning rate version 4')
    #ax[0].set_title('Example of a high initial learning rate')
    #ax[0].set_title('Re-training using ALRGD')
    ax[0].set_title(args.title)
    
    #outfile = filepath + '.png'
    #outfile = filepath + '.svg'
    outfile = os.path.basename(filepath) + '.pgf'
    plt.savefig(outfile) #save to file
    
    outfile = os.path.basename(filepath) + '.png'
    plt.savefig(outfile) #save again
    plt.show()
    #plt.close(fig)
    
def plotversus(args):
    filepaths = [args.file, args.overlay]
  
    fig, ax = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[2, 1, 1]},figsize=(6,8))
    for i, filepath in enumerate(filepaths):
      color = '#1f77b4'
      label='ALRGD'
      if i==1:
        color = '#ff7f0e'#'#17becf'
        label='SGD'
      stats = getdata(filepath)
      stats['training'] = [i * 100 for i in stats['training']]
      stats['validation'] = [i * 100 for i in stats['validation']]
      ax[0].plot(stats['steps'], stats['validation'], label=label, color=color)
      ax[1].plot(stats['steps'], stats['learning_rate'],'g-', label='learning_rate', color=color)
      ax[2].plot(stats['steps'], stats['cross_entropy'],'r-', label='cross_entropy', color=color)
      
    ax[0].set_ylabel('validation accuracy in %')
    ax[2].set_xlabel('epoch')
    ax[0].legend(loc='lower right')
    ax[1].set_ylabel('learning rate')
    ax[2].set_ylabel('cross entropy')
    ax[0].set_title('ALRGD vs SGD')
    
    #outfile = filepath + '.png'
    #outfile = filepath + '.svg'
    n1 = os.path.basename(filepaths[0])
    n2 = os.path.basename(filepaths[1])
    name = n1+'_vs_'+n2
    outfile = name + '.pgf'
    #plt.savefig(outfile) #save to file
    
    outfile = name + '.png'
    plt.savefig(outfile) #save again
    plt.show()
    #plt.close(fig)
    
def plotversusada(args):
    filepaths = [args.file, args.overlay]
  
    fig, ax = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[2, 1]},figsize=(6,6))
    for i, filepath in enumerate(filepaths):
      color = '#1f77b4'
      label='ALRGD'
      if i==1:
        color = '#ff7f0e'#e377c2'#d62728'
        label='AdaDelta'
      stats = getdata(filepath)
      stats['training'] = [i * 100 for i in stats['training']]
      stats['validation'] = [i * 100 for i in stats['validation']]
      ax[0].plot(stats['steps'][:500], stats['validation'][:500], label=label, color=color)
      ax[1].plot(stats['steps'][:500], stats['cross_entropy'][:500],'r-', label='cross_entropy', color=color)
      
    ax[0].set_ylabel('validation accuracy in %')
    ax[1].set_xlabel('epoch')
    ax[0].legend(loc='lower right')
    ax[1].set_ylabel('cross entropy')
    ax[0].set_title('ALRGD vs AdaDelta')
    
    #outfile = filepath + '.png'
    #outfile = filepath + '.svg'
    n1 = os.path.basename(filepaths[0])
    n2 = os.path.basename(filepaths[1])
    name = n1+'_vs_'+n2
    outfile = name + '.pgf'
    plt.savefig(outfile) #save to file
    
    outfile = name + '.png'
    plt.savefig(outfile) #save again
    plt.show()
    #plt.close(fig)
    
def plotversusadamulti(args):
    filepaths = [args.file, args.overlay]
    
    if args.hidelr:
      fig, ax = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[2, 1]},figsize=(6,6))
      crax = ax[1]
    else:
      fig, ax = plt.subplots(3, sharex=True, gridspec_kw = {'height_ratios':[2, 1, 1]},figsize=(6,8))
      crax = ax[2]
    axu2 = crax
    if args.splitcr:
      axu2 = crax.twinx()
    axu0 = ax[0]
    if args.splitacc:
      axu0 = ax[0].twinx()
    axtwins=[axu0,axu2]
    lns = None
    for i, filepath in enumerate(filepaths):
      color = '#1f77b4'
      label='ALRGD'
      axu = [ax[0],crax]
      if i==1:
        color = '#ff7f0e'#e377c2'#d62728'
        label='AdaDelta'
        axu = axtwins
      stats = getdata(filepath)
      stats['training'] = [i * 100 for i in stats['training']]
      stats['validation'] = [i * 100 for i in stats['validation']]
      ln = axu[0].plot(stats['steps'][:500], stats['validation'][:500], label=label, color=color)
      if lns:
        lns += ln
      else:
        lns = ln
      axu[1].plot(stats['steps'][:500], stats['cross_entropy'][:500],'r-', label='cross_entropy', color=color)
      if i==0 and not args.hidelr:
        ax[1].plot(stats['steps'], stats['learning_rate'],'g-', label='learning_rate', color=color)
      elif args.limitacc:
        axu[0].set_ylim(axu[0].get_ylim()[0], float(args.limitacc)) 
      if args.splitacc:
        axu[0].tick_params('y', colors=color)
      if args.splitcr:
        axu[1].tick_params('y', colors=color)
    
    labs = [l.get_label() for l in lns]
      
    ax[0].set_ylabel('validation accuracy in %')
    ax[1].set_xlabel('epoch')
    ax[0].legend(lns, labs,loc='lower right')
    if not args.hidelr:
      ax[1].set_ylabel('learning rate')
    crax.set_ylabel('cross entropy')
    ax[0].set_title(args.title)
    
    #outfile = filepath + '.png'
    #outfile = filepath + '.svg'
    n1 = os.path.basename(filepaths[0])
    n2 = os.path.basename(filepaths[1])
    name = n1+'_vs_'+n2
    outfile = name + '.pgf'
    plt.savefig(outfile) #save to file
    
    outfile = name + '.png'
    plt.savefig(outfile) #save again
    plt.show()
    #plt.close(fig)

def show(args):
    filepath = args.file
    stats = getdata(filepath)
    print("Evaluation accuracy: %.1f%%" % (stats['evaluation']*100))
    print("Epochs %d" % len(stats['training']))



def main(args):

  if args.type:
    tdict = {"lrcomp": plotlrcomp,"alrgd": plotalrgd,"versus": plotversus,"versusada": plotversusada,"multiscale": plotversusadamulti,"show": show}
    tdict[args.type](args)

  elif args.file:
    if args.overlay:
      plotdouble(args.file, args.overlay, True)
    else:
      plotsingle(args.file, True)
      
  elif args.input_dir:
    plotmultiple(args.input_dir)
    
  elif args.dir:
    savemultiple(args.dir)
    
  else:
    print ("Please provide file or directory.")
  
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--type',
      type=str,
      help='The type of plot to create.'
  )
  parser.add_argument(
      '--file',
      type=str,
      help='Path to the file to visualize.'
  )
  parser.add_argument(
      '--overlay',
      type=str,
      help='Path to the file to include in the graph.'
  )
  parser.add_argument(
      '--input_dir',
      type=str,
      help='Path to the directory to load statistics from.'
  )
  parser.add_argument(
      '--dir',
      type=str,
      help='Path to the directory to load statistics from.'
  )
  parser.add_argument(
      '--limitacc',
      type=str,
      default=None,
      help='limit axis of accuracy.'
  )
  parser.add_argument(
      '--title',
      type=str,
      default=None,
      help='limit axis of accuracy.'
  )
  parser.add_argument(
      '--splitcr',
      default=False,
      help='mode for multiscale.',
      action='store_true'
  )
  parser.add_argument(
      '--splitacc',
      default=False,
      help='mode for multiscale.',
      action='store_true'
  )
  parser.add_argument(
      '--hidelr',
      default=False,
      help='mode for multiscale.',
      action='store_true'
  )
  args = parser.parse_args()
  main(args)
