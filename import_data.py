"""Functions for downloading and reading MNIST data."""
# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import array
import tensorflow as tf
import csv
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


#training data is one file per utterance of form:
#label
#[1,2,3,4,...,x] - spectrogram for nth window
#[1,2,3,4,...,x] - logs
#1				 - energy
#[1,2,3,4,...,x] - standard histogram
#*				 - divider
#etc
def read_training_data(filename,n_steps,training=True):
	if training == True:
		print("reading file: ",filename)
	spectrogram = []
	logs = []
	energy = []
	histogram = []
	revolve = 0
	first = True
	label = ""
	with open(filename, 'rb') as csvfile:
		data = csv.reader(csvfile, delimiter=',') 
		for row in data: #every row is a row of the csv. without formatting each array looks like ['1','2',...,'n'] #data prints the csvreader object
						 #every row should be an array of time slices for a single utterance
			if training==True:
				if first:
					label = row
					first = False
				elif revolve == 0:
					revolve+=1
				elif revolve == 1:
					p = [s.strip('[]') for s in row]
					p = [float(s) for s in p]
					spectrogram.append(p)
					revolve+=1
				elif revolve == 2:
					l = [s.strip('[]') for s in row]
					l = [float(s) for s in l]
					logs.append(l)
					revolve+=1
				elif revolve == 3:
					e = [s.strip('[]') for s in row]
					e = [float(s) for s in e]
					energy.append(e)
					revolve += 1
				elif revolve == 4:
					h = [s.strip('[]') for s in row]
					h = [float(s) for s in h]
					histogram.append(h)
					revolve = 0
				else:
					revolve = 1;
			else:
				label = ""
				if revolve == 0:
					p = [s.strip('[]') for s in row]
					p = [float(s) for s in p]
					spectrogram.append(p)
					revolve+=1
				elif revolve == 1:
					l = [s.strip('[]') for s in row]
					l = [float(s) for s in l]
					logs.append(l)
					revolve+=1
				elif revolve == 2:
					e = [s.strip('[]') for s in row]
					e = [float(s) for s in e]
					energy.append(e)
					revolve += 1
				elif revolve == 3:
					h = [s.strip('[]') for s in row]
					h = [float(s) for s in h]
					histogram.append(h)
					revolve +=1
				else:
					revolve = 0;
				
	csvfile.close()
	test = np.array(spectrogram)
	#test = np.array(logs)
	test.resize(n_steps,160) #160 for 16000hz, can be changed
	#print(test.shape)
	return label,test
	

def read_spectrogram(filename):
	print("Reading Spectrogram: ",filename)
	out = []
	with open(filename, 'rb') as csvfile:
		data = csv.reader(csvfile, delimiter='[',lineterminator=']') 
		for row in data: #every row is a row of the csv. without formatting each array looks like ['1','2',...,'n'] #data prints the csvreader object
						 #every row should be an array of time slices for a single utterance
			out.append(row)
	#print(out)
	return out

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


#start is the column which is before the start of the label, in our case for ted-lium data the transcript begins on the 7th column
def read_labels(filename, start=6):
	print("Reading Label: ",filename)
	out = []
	with open(filename, 'rb') as csvfile:
		data = csv.reader(csvfile, delimiter=' ')
		for row in data: #every row should be a string which represents the label
			out.append(" ".join(row[start:]))
	out = np.asarray(out)
	return out



def read_dir(train_dir, batch_size,n_steps, training=True):
	speech = []
	labels = []
	i=0
	files = [f for f in listdir(train_dir) if isfile(train_dir+'/'+f)]
	for file in files:
		''' 
		#Causes index out of bounds errors
		a,b = read_training_data(train_dir+'/'+file,n_steps, training)
		test = np.array(b)
		labels.append(a)
		speech.append(b)
		'''
		if i<batch_size:
			a,b = read_training_data(train_dir+'/'+file,n_steps, training)
			test = np.array(b)
			#print(test.shape)
			labels.append(a)
			speech.append(b)
			i+=1
		else:
			break
		
	
#	speechfiles = [f for f in listdir(speech_path) if isfile(join(speech_path, f))]
#	for file in speechfiles:
#		speech.append(read_training_data(speech_path+file))
	#print(len(speech))
	#print(len(speech[0]))
	#print(len(speech[0][0]))
	speech = np.array(speech)
	#speech.shape=(10,1)
	#speech.shape = (len(speech),len(speech[0]), len(speech[0][0]))
	#print(speech.shape)
	#speech.shape = (data_size,num_windows,spectrogram_size)
#	
#	labelfiles = [f for f in listdir(label_path) if isfile(join(label_path, f))]
#	for file in labelfiles:
#		labels.append(read_labels(label_path+file))
	labels = np.array(labels)	
	#print(speech)
	return speech,labels
	
class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],
      #                       images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(np.float32)
        #images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, batch_size, n_steps, training=True,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  train_speech,train_labels = read_dir(train_dir, batch_size,n_steps, training)
  

  train = DataSet(train_speech, train_labels, dtype=dtype)

  return train


def load_mnist():
  return read_data_sets('MNIST_data')
  
#if __name__ == '__main__':
  #read_spectrogram("test.txt")
  #read_labels("911Mothers_2010W.stm",6)
  #read_data_sets("train",10,5000)