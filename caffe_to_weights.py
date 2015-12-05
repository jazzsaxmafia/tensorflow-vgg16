import os
os.environ["GLOG_minloglevel"] = "2"

import ipdb
from utils import *
import matplotlib.pyplot as plt
import skimage
import caffe
import numpy as np
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('only_caffe', False,
                            """Only run caffe""")
tf.app.flags.DEFINE_boolean('only_tf', False,
                            """Only run tf""")

def tf_show_layer(image):
  skimage.io.imshow(image)
  skimage.io.show()

# input as gotten from skimage.io.imread() that is
# [height, width, 3] and scaled between 0 and 1
# output is scaled to 0 - 255 with mean subtracted
# output [in_channels, in_height, in_width]
def preprocess(img):
  out = np.copy(img) * 255
  out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
  # sub mean
  out[:,:,0] -= VGG_MEAN[0]
  out[:,:,1] -= VGG_MEAN[1]
  out[:,:,2] -= VGG_MEAN[2]
  out = out.transpose((2,0,1)) # h, w, c -> c, h, w
  return out

def deprocess(img):
  out = np.copy(img)
  out = out.transpose((1,2,0)) # c, h, w -> h, w, c

  out[:,:,0] += VGG_MEAN[0]
  out[:,:,1] += VGG_MEAN[1]
  out[:,:,2] += VGG_MEAN[2]
  out = out[:, :, [2,1,0]]
  out /= 255
  return out

#caffe.set_mode_cpu()
net_caffe = caffe.Net("VGG_2014_16.prototxt", "VGG_ILSVRC_16_layers.caffemodel", caffe.TEST)


caffe_layers = {}
caffe_layers_numpy = {}

for i, layer in enumerate(net_caffe.layers):
    layer_name = net_caffe._layer_names[i]
    caffe_layers[layer_name] = layer

def caffe_to_numpy(layer_name):
    layer = caffe_layers[layer_name]
    if len(layer.blobs) > 0:
        W = layer.blobs[0].data
        b = layer.blobs[1].data
        return [W,b]
    else:
        return None

def caffe_weights(layer_name):
    layer = caffe_layers_numpy[layer_name]
    return layer[0]#.blobs[0].data

def caffe_bias(layer_name):
    layer = caffe_layers_numpy[layer_name]
    return layer[1]#layer.blobs[1].data

for k in caffe_layers.iterkeys():
    caffe_layers_numpy[k] = caffe_to_numpy(k)

with open("caffe_layers_value.pickle", "w") as f:
    cPickle.dump(caffe_layers_numpy, f)
