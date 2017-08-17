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
import fnmatch
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix

  
def getdata(filepath):
  data = []
  
  with open(filepath, "rb") as picklefile:
    stats = pickle.load(picklefile)
    data = stats
  
  return data


def getconfusionmatrix(data):
  numcl = len(data['labels'])
  
  #use tensorflow (gives tensor)
  # tf.contrib.metrics.confusion_matrix(data['truths'], data['predictions'],numcl)

  #use scikit
  return confusion_matrix(data['truths'], data['predictions'])
  
def printLatex(m, hrlabels):
  hrlabels = [l.replace("&", "\&") for l in hrlabels]
  rows = m
  rows = [[hrlabels[i]]+row for i, row in enumerate(rows)]
  rows.insert(0, ["", ""]+hrlabels)
  clinestr = "\\cline{2-%d}"%(len(m)+2)
  table = (" \\\\\n"+clinestr+"\n&").join([" & ".join(map(str,line)) for line in rows])
  
  print("\\begin{tabular}{| c |"+ (" c |"*(len(hrlabels)+1))+"}")
  print("\\hline\n&\\multicolumn{%d}{c|}{Prediction}\\\\"%(len(m)+1))
  print("\\hline")
  print("\\parbox[t]{1ex}{\\multirow{%d}{*}{\\rotatebox[origin=c]{90}{Actual}}}"%(len(m)+1))
  print(table)
  print("\\\\\\hline")
  print("\\end{tabular}")
  

def main(args):

  if not args.file:
    print ('pls specify file')
    
  data = getdata(args.file)
  
  #print(data['predictions'])
  #print(data['truths'])
  print(data['labels'])
  if(args.labels):
    hrlabelorig = [line.rstrip('\n') for line in open(args.labels)]
  
    for li in data['labels']:
      lint = int(li)
      print(hrlabelorig[lint])
      
    hrlabels = [hrlabelorig[int(li)] for li in data['labels']]
    #print(hrlabels)
  
  #for i, p in enumerate(data['predictions']):
  #  if not p == data['truths'][i]:
  #    print(p)
  
  m = getconfusionmatrix(data)
  print(m)
  
  # now in percent
  mper = [[p/sum(truth) for p in truth] for truth in m]
  #print(mper)
  # colors assume that only exactly the diagonal has >50%
  #mperprint = [["%.1f \cellcolor[rgb]{%.2f,%.2f,%.2f}"%(x*100,1 if x<0.5 else 0,1-x*4 if x<0.5 else x,1-x*4 if x<0.5 else 0) for x in y] for y in mper]
  mperprint = [["%.1f \cellcolor[rgb]{%.2f,%.2f,%.2f}"%(x*100,1 if x<0.5 else 1-(x-0.5),1-x*4 if x<0.5 else 1,1-x*4 if x<0.5 else 1-(x-0.5)) for x in y] for y in mper] 
  #[["%.1f\\%%"%(x*100) for x in y] for y in mper]
  
  # prepare latex table
  printTable = True
  if printTable:
    #printLatex(m.tolist(),hrlabels)
    print('')
    printLatex(mperprint,hrlabels)
  
  
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--file',
      type=str,
      help='Path to the file to create confusion matrix from.'
  )
  parser.add_argument(
      '--labels',
      type=str,
      help='Path to the file with readable labels.'
  )
  args = parser.parse_args()
  main(args)
