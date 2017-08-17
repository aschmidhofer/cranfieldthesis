# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
#
# This file was modified by Andreas Schmidhofer, July 2017
#
# (most modification contain a comment tagged with AS)
#
# ==============================================================================

"""Simple transfer learning with an Inception v3 architecture model.

With support for TensorBoard.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.


To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

tensorboard --logdir /tmp/retrain_logs

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import pickle #by AS

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'#'http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz'
          
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MAX_STEPS = 100000
DYNAMIC_STEPS_LOOKBACK = 16 # used by stopping criteria
CHANGING_LEARNING_RATE_ADJUST_PERCENTAGE_MODIFYER = 10.0 # changes the amount of \zeta depending on cross entropy difference
CHANGING_LEARNING_RATE_MIN = 1.2 # the static amount of \zeta (must be >= 1.0)
MIN_LEARNING_RATE = 0.001
OPTIMIZERS = ['ALRGD'] # list of optimizers to run

def create_image_lists(image_dir): # note: percentages not supported anymore
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  # AS - this part was changed to support a directory per category
  for cat in ['training', 'testing', 'validation']:
    category_dir = os.path.join(image_dir,cat)
    if not gfile.Exists(category_dir):
      print("Category directory '" + category_dir + "' not found.")
      return None
    
    sub_dirs = [x[0] for x in gfile.Walk(category_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
      if is_root_dir:
        is_root_dir = False
        continue
      extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
      file_list = []
      dir_name = os.path.basename(sub_dir)
      if dir_name == category_dir:
        continue
      print("Looking for images in '" + dir_name + "'")
      for extension in extensions:
        file_glob = os.path.join(category_dir, dir_name, '*.' + extension)
        file_list.extend(gfile.Glob(file_glob))
      if not file_list:
        print('No files found')
        continue
      if len(file_list) < 20:
        print('WARNING: Folder has less than 20 images, which may cause issues.')
      elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
        print('WARNING: Folder {} has more than {} images. Some images will '
              'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
      label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
      those_images = []
      for file_name in file_list:
        base_name = os.path.basename(file_name)
        those_images.append(base_name)
      
      if not(label_name in result):
        result[label_name] = {
          'dir': dir_name,
          'training': [],
          'testing': [],
          'validation': [],
        }
      result[label_name][cat] = those_images
  return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = os.path.join(category, label_lists['dir']) # AS changed this line to support a directory per category
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'


def create_inception_graph():
  with tf.Graph().as_default() as graph:
    model_filename = os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract():
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def write_list_of_floats_to_file(list_of_floats, file_path):
  s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
  with open(file_path, 'wb') as f:
    f.write(s)


def read_list_of_floats_from_file(file_path):

  with open(file_path, 'rb') as f:
    s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
    return list(s)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  print('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, bottleneck_tensor)
  except:
    raise RuntimeError('Error during processing file %s' % image_path)

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
  label_lists = image_lists[label_name]
  sub_dir = os.path.join(category,label_lists['dir']) # AS changed to support dir per cat
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(sess, image_lists, label_name, index,
                                 image_dir, category, bottleneck_dir,
                                 jpeg_data_tensor, bottleneck_tensor)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          print(str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                            image_index, image_dir, category,
                                            bottleneck_dir, jpeg_data_tensor,
                                            bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                              image_index, image_dir, category,
                                              bottleneck_dir, jpeg_data_tensor,
                                              bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames




def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, optimizer_name, my_learning_rate):
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count],
                                          stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    if(optimizer_name=='ALRGD'):
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=my_learning_rate)
      train_step = optimizer.minimize(cross_entropy_mean)
    if(optimizer_name=='ALRGD_adam'):
      optimizer = tf.train.AdamOptimizer(learning_rate=my_learning_rate)
      train_step = optimizer.minimize(cross_entropy_mean)
    elif(optimizer_name=='gradientdescent'):
      optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
      train_step = optimizer.minimize(cross_entropy_mean)
    elif (optimizer_name=='adagrad'):
      optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate) 
      train_step = optimizer.minimize(cross_entropy_mean)
    elif (optimizer_name=='adam'):
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate) 
      train_step = optimizer.minimize(cross_entropy_mean)
    elif (optimizer_name=='ftrl'):
      optimizer = tf.train.FtrlOptimizer(FLAGS.learning_rate) 
      train_step = optimizer.minimize(cross_entropy_mean)
    elif (optimizer_name=='adadelta'):
      optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate) 
      train_step = optimizer.minimize(cross_entropy_mean)
      

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def retrain(image_lists, optimizer_name=OPTIMIZERS[0], save_graph=True, adjustLR=True):

  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())
      
      
  with graph.as_default():
    my_learning_rate = tf.placeholder(tf.float32, shape=[],name='learning_rate_input') # AS

    with tf.Session(graph=graph) as sess:

      # Add the new layer that we'll be training.
      (train_step, cross_entropy, bottleneck_input, ground_truth_input,
       final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                              FLAGS.final_tensor_name,
                                              bottleneck_tensor, optimizer_name, my_learning_rate)

      # Create the operations we need to evaluate the accuracy of our new layer.
      evaluation_step, prediction = add_evaluation_step(
          final_tensor, ground_truth_input)

      # Merge all the summaries and write them out to the summaries_dir
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                           sess.graph)

      validation_writer = tf.summary.FileWriter(
          FLAGS.summaries_dir + '/validation')

      # Set up all our weights to their initial default values.
      init = tf.global_variables_initializer()
      sess.run(init)
      
      # AS - prepare saver
      saver = tf.train.Saver()
      randomID = random.randint(0,9999999)
      base_ckpt = '/tmp/'+str(randomID)+'_base.ckpt'
      best_ckpt = '/tmp/'+str(randomID)+'_best.ckpt'

      # AS - initial learning rate
      myLR = FLAGS.learning_rate
      
      # AS - keeping my own stats
      mystat_steps = []
      mystat_training = []
      mystat_validation = []
      mystat_learningrate = []
      mystat_crossentropy = []
      
      best_train_accuracy = 0
      best_cross_entropy = 0 # well...
      
      steps = FLAGS.how_many_training_steps
      dynamicsteps = (FLAGS.how_many_training_steps == -1)
      if dynamicsteps:
        steps = MAX_STEPS

      # Run the training for as many cycles as requested on the command line.
      for i in range(steps):
        print(i)
      
        # AS - check if stopping criterion is reached
        if dynamicsteps:
          LB = DYNAMIC_STEPS_LOOKBACK
          if i>LB:
            #avg = np.mean(mystat_validation[-LB:-1])
            #if mystat_validation[-1]<avg:
            compare = np.min(mystat_validation[-LB:-1])
            if mystat_validation[-1]<=compare:
              saver.restore(sess, base_ckpt) # previous was better
              break
      
      
      
        def dotraining(learning_rate_to_use_this_step):
          STEPS = 1 # TODO increase when using smaller batch sizes
          for stp in range(STEPS): 
              
            # Get a batch of input bottleneck values from the cache stored on disk.
            (train_bottlenecks,
             train_ground_truth, _) = get_random_cached_bottlenecks(
                 sess, image_lists, FLAGS.train_batch_size, 'training',
                 FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                 bottleneck_tensor)
            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.

            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth,
                           my_learning_rate: learning_rate_to_use_this_step}) 
            #train_writer.add_summary(train_summary, i)
        
        def dotrainingeval():
          (train_bottlenecks,
           train_ground_truth, _) = get_random_cached_bottlenecks(
               sess, image_lists, -1, 'training',
               FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
               bottleneck_tensor)
          return sess.run(
              [evaluation_step, cross_entropy],
              feed_dict={bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth})
         
        # save state
        saver.save(sess, base_ckpt)
        
        # AS - adjusting learning rates
        if(adjustLR):
          previous_best_train_accuracy = best_train_accuracy
          previous_best_cross_entropy = best_cross_entropy
          
          # same LR
          dotraining(myLR)
          sameLR_train_accuracy, sameLR_cross_entropy = dotrainingeval()
          saver.save(sess, best_ckpt)
          best_train_accuracy = sameLR_train_accuracy
          best_cross_entropy = sameLR_cross_entropy
          print('%s: Same LR Train accuracy = %.1f%%, cross entropy = %f' % (datetime.now(), sameLR_train_accuracy * 100, sameLR_cross_entropy))
          
          # AS - check improvement
          if (sameLR_cross_entropy>previous_best_cross_entropy):
            # recalculate learning rate
            initialstep = (previous_best_cross_entropy==0.0)
            if not initialstep:
              print('%s: Warning! cross entropy increased from %f to %f' % (datetime.now(), previous_best_cross_entropy, sameLR_cross_entropy))
            currentLR = myLR
            previous_cross_entropy = sameLR_cross_entropy
            while (best_cross_entropy>previous_best_cross_entropy):
              currentLR = currentLR / 2 # decrease lr exponentially
              if(currentLR<MIN_LEARNING_RATE): break
              saver.restore(sess, base_ckpt)
              dotraining(currentLR)
              current_train_accuracy, current_cross_entropy = dotrainingeval()
              print('%s: LR = %f, Train accuracy = %.1f%%, cross entropy = %f' % (datetime.now(), currentLR, current_train_accuracy * 100, current_cross_entropy))
              if(current_cross_entropy<=previous_best_cross_entropy):
                myLR = currentLR
                best_train_accuracy = current_train_accuracy
                best_cross_entropy = current_cross_entropy
                saver.save(sess, best_ckpt)
              elif(initialstep):
                if(current_cross_entropy>previous_cross_entropy):
                  saver.restore(sess, best_ckpt)
                  break
                else:
                  previous_cross_entropy = current_cross_entropy
                  myLR = currentLR
                  best_train_accuracy = current_train_accuracy
                  best_cross_entropy = current_cross_entropy
                  saver.save(sess, best_ckpt)
                
          else: # go on as normal (cross entropy did not increase)
            
              
            # AS - calculate modifyer
            c = CHANGING_LEARNING_RATE_ADJUST_PERCENTAGE_MODIFYER
            cross_entropy_diff = previous_best_cross_entropy-sameLR_cross_entropy
            cross_entropy_diff_perc = abs(cross_entropy_diff/sameLR_cross_entropy)
            cmin = CHANGING_LEARNING_RATE_MIN
            
            lrmodify = cmin+(c*cross_entropy_diff_perc)
            
            highLR = myLR*lrmodify
            lowLR = myLR/lrmodify
            #hugeLR = myLR*20
            #tinyLR = myLR/20
            
            print('%s: LR modifyer %.3f: %3f and %3f' % (datetime.now(), lrmodify, lowLR, highLR))
            
            
            # low
            saver.restore(sess, base_ckpt)
            dotraining(lowLR)
            lowLR_train_accuracy, lowLR_cross_entropy = dotrainingeval()
            print('%s: Low LR Train accuracy = %.1f%%, cross entropy = %f' % (datetime.now(), lowLR_train_accuracy * 100, lowLR_cross_entropy))
            #if (lowLR_train_accuracy>compare_train_accuracy):
            if(lowLR_cross_entropy<best_cross_entropy):
              myLR = lowLR
              best_train_accuracy = lowLR_train_accuracy
              compare_train_accuracy = best_train_accuracy
              best_cross_entropy = lowLR_cross_entropy
              saver.save(sess, best_ckpt)
              
              
            # high
            saver.restore(sess, base_ckpt)
            dotraining(highLR)
            highLR_train_accuracy, highLR_cross_entropy = dotrainingeval()
            print('%s: High LR Train accuracy = %.1f%%, cross entropy = %f' % (datetime.now(), highLR_train_accuracy * 100, highLR_cross_entropy))
            #if (highLR_train_accuracy>compare_train_accuracy):
            if(highLR_cross_entropy<best_cross_entropy):
              myLR = highLR
              best_train_accuracy = highLR_train_accuracy
              compare_train_accuracy = best_train_accuracy
              best_cross_entropy = highLR_cross_entropy
              #saver.save(sess, best_ckpt)
            else:
              saver.restore(sess, best_ckpt) 
              
          
          
          print ("%s: proceeding with LR = %f"%(datetime.now(),myLR))
        
        else:
          dotraining(myLR)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
          train_accuracy, cross_entropy_value = dotrainingeval()
          print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                          train_accuracy * 100))
          print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                     cross_entropy_value))
          validation_bottlenecks, validation_ground_truth, _ = (
              get_random_cached_bottlenecks(
                  sess, image_lists, FLAGS.validation_batch_size, 'validation',
                  FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                  bottleneck_tensor))
          # Run a validation step and capture training summaries for TensorBoard
          # with the `merged` op.
          validation_summary, validation_accuracy = sess.run(
              [merged, evaluation_step],
              feed_dict={bottleneck_input: validation_bottlenecks,
                         ground_truth_input: validation_ground_truth})
          validation_writer.add_summary(validation_summary, i)
          print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                (datetime.now(), i, validation_accuracy * 100,
                 len(validation_bottlenecks)))
                 
          mystat_steps.append(i)
          mystat_training.append(train_accuracy)
          mystat_validation.append(validation_accuracy)
          mystat_learningrate.append(myLR)
          mystat_crossentropy.append(cross_entropy_value)

      # We've completed all our training, so run a final test evaluation on
      # some new images we haven't used before.
      test_bottlenecks, test_ground_truth, test_filenames = (
          get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                        'testing', FLAGS.bottleneck_dir,
                                        FLAGS.image_dir, jpeg_data_tensor,
                                        bottleneck_tensor))
      test_accuracy, predictions = sess.run(
          [evaluation_step, prediction],
          feed_dict={bottleneck_input: test_bottlenecks,
                     ground_truth_input: test_ground_truth})
      print('Final test accuracy = %.1f%% (N=%d)' % (
          test_accuracy * 100, len(test_bottlenecks)))

      # AS - prepare for confusion matrix
      labels = []
      for i, test_filename in enumerate(test_filenames):
        labels.append(test_ground_truth[i].argmax())

      if FLAGS.print_misclassified_test_images:
        print('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
          if predictions[i] != test_ground_truth[i].argmax():
            print('%70s  %s' % (test_filename,
                                list(image_lists.keys())[predictions[i]]))
      
      if (save_graph):
        # Write out the trained graph and labels with the weights stored as
        # constants.
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
        with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
          f.write(output_graph_def.SerializeToString())
        with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
          f.write('\n'.join(image_lists.keys()) + '\n')
          
      # AS - return statistics
      mystat = {
          "steps": mystat_steps,
          "training": mystat_training,
          "validation": mystat_validation,
          "learning_rate": mystat_learningrate,
          "cross_entropy": mystat_crossentropy,
          "evaluation": test_accuracy,
          "predictions": predictions,
          "truths": labels, 
          "labels": list(image_lists.keys())
          }
      return mystat

def main(_):
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.image_dir)
  class_count = len(image_lists.keys())
  if class_count == 0:
    print('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    print('Only one valid folder of images found at ' + FLAGS.image_dir +
          ' - multiple classes are needed for classification.')
    return -1

      
  # Set up the pre-trained graph.
  maybe_download_and_extract()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())
      
      
   # AS - if user is sure bottlenecks exist this part can be skipped
  bottlenecks_exist = FLAGS.bottlenecks_exist
  if not bottlenecks_exist:
    with tf.Session(graph=graph) as sess:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        bottleneck_tensor)
      
  # AS - run retrainings for each optimizer
  optims = OPTIMIZERS
  lr = FLAGS.learning_rate
  if FLAGS.optimizer:
    optims = [FLAGS.optimizer]
  for optim in optims:
    adjustLR = optim.startswith('ALRGD')
    lrs = str(lr).replace('.', 'p')
    vers='vT' # version string for output file
    stepsstr = str(FLAGS.how_many_training_steps)
    if(FLAGS.how_many_training_steps == -1):
      stepsstr = 'dynamic'
    filename = optim+'_'+vers+'_'+lrs+'_'+stepsstr+'.data'
    path = os.path.join(FLAGS.output_dir, filename)
    if os.path.exists(path):
      print(filename + " already exists. - skip")
      continue
    else: 
      print("working on "+optim+ " with lr="+str(lr)+" "+stepsstr+" steps.")
    
    with open(path, "wb") as picklefile:
      mystats = retrain(image_lists,optim,True,adjustLR)
      pickle.dump(mystats, picklefile)
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default='',
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='/tmp/output_graph.pb',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=4000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,#0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=-1,#100, 
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=-1,#100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--bottlenecks_exist',
      default=False,
      help="""\
      Whether to skip the bottleneck creation step.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--output_dir',
      type=str,
      default='.',
      help='Path to the directory to store training statistics.'
  )
  parser.add_argument(
      '--optimizer',
      type=str,
      help="Select the optimizer to use. Possible values are ['ALRGD', 'ALRGD_adam', 'adadelta', 'gradientdescent', 'adagrad', 'adam', 'ftrl'] "
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
