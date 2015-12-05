import os
os.environ["GLOG_minloglevel"] = "2"

import cPickle
import ipdb
from utils import *
import matplotlib.pyplot as plt
import skimage
import caffe
import numpy as np
import tensorflow as tf

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

with open("caffe_layers_value.pickle") as f:
    print "Loading VGG weights ... "
    caffe_layers_numpy = cPickle.load(f)
    print "Done loading ..."


def caffe_weights(layer_name):
    layer = caffe_layers_numpy[layer_name]
    return layer[0]#.blobs[0].data

def caffe_bias(layer_name):
    layer = caffe_layers_numpy[layer_name]
    return layer[1]#layer.blobs[1].data

# converts caffe filter to tf
# tensorflow uses [filter_height, filter_width, in_channels, out_channels]
#                  2               3            1            0
# need to transpose channel axis in the weights
# caffe:  a convolution layer with 96 filters of 11 x 11 spatial dimension
# and 3 inputs the blob is 96 x 3 x 11 x 11
# caffe uses [out_channels, in_channels, filter_height, filter_width]
#             0             1            2              3
def caffe2tf_filter(name):
  f = caffe_weights(name)
  return f.transpose((2, 3, 1, 0))

def conv_layer(m, bottom, name):
  with tf.variable_scope(name) as scope:
    w = caffe2tf_filter(name)
    #print name + " w shape", w.shape
    conv_weight = tf.constant(w, dtype=tf.float32, name="filter")

    conv = tf.nn.conv2d(bottom, conv_weight, [1, 1, 1, 1], padding='SAME')

    conv_biases = tf.constant(caffe_bias(name), dtype=tf.float32, name="biases")
    bias = tf.nn.bias_add(conv, conv_biases)

    if name == "conv1_1":
      m["conv1_1_weights"] = conv_weight
      m["conv1_1a"] = bias

    relu = tf.nn.relu(bias, name=name)
    return relu

def fc_layer(bottom, name):
  shape = bottom.get_shape().as_list()
  dim = 1
  for d in shape[1:]:
     dim *= d
  x = tf.reshape(bottom, [-1, dim])

  print name, "caffe weight shape", caffe_weights(name).shape

  cw = caffe_weights(name)
  if name == "fc6":
    assert cw.shape == (4096, 25088)
    cw = cw.reshape((4096, 512, 7, 7))
    cw = cw.transpose((2, 3, 1, 0))
    cw = cw.reshape(25088, 4096)
  else:
    cw = cw.transpose((1, 0))

  weights = tf.constant(cw, dtype=tf.float32)
  biases = tf.constant(caffe_bias(name), dtype=tf.float32)

  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  fc = tf.nn.bias_add(tf.matmul(x, weights), biases, name=name)

  return fc

# Input should be an rgb image [batch, height, width, 3]
# values scaled [0, 1]
def inference(rgb):
  m = {}

  rgb_scaled = rgb * 255.0

  # Convert RGB to BGR
  red, green, blue = tf.split(3, 3, rgb_scaled)
  assert red.get_shape().as_list()[1:] == [224, 224, 1]
  assert green.get_shape().as_list()[1:] == [224, 224, 1]
  assert blue.get_shape().as_list()[1:] == [224, 224, 1]
  bgr = tf.concat(3, [
    blue - VGG_MEAN[0],
    green - VGG_MEAN[1],
    red - VGG_MEAN[2],
  ])
  assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

  relu1_1 = conv_layer(m, bgr, "conv1_1")
  relu1_2 = conv_layer(m, relu1_1, "conv1_2")
  pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  m["relu1_1"] = relu1_1

  relu2_1 = conv_layer(m, pool1, "conv2_1")
  relu2_2 = conv_layer(m, relu2_1, "conv2_2")
  pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  relu3_1 = conv_layer(m, pool2, "conv3_1")
  relu3_2 = conv_layer(m, relu3_1, "conv3_2")
  relu3_3 = conv_layer(m, relu3_2, "conv3_3")
  pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  relu4_1 = conv_layer(m, pool3, "conv4_1")
  relu4_2 = conv_layer(m, relu4_1, "conv4_2")
  relu4_3 = conv_layer(m, relu4_2, "conv4_3")
  pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4')

  relu5_1 = conv_layer(m, pool4, "conv5_1")
  relu5_2 = conv_layer(m, relu5_1, "conv5_2")
  relu5_3 = conv_layer(m, relu5_2, "conv5_3")
  pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5')

  m['pool5'] = pool5

  print "pool5 shape", pool5.get_shape().as_list()

  fc6 = fc_layer(pool5, "fc6")
  assert fc6.get_shape().as_list() == [None, 4096]
  m['fc6a'] = fc6

  relu6 = tf.nn.relu(fc6)
  drop6 = tf.nn.dropout(relu6, 0.5)

  fc7 = fc_layer(drop6, "fc7")
  relu7 = tf.nn.relu(fc7)
  drop7 = tf.nn.dropout(relu7, 0.5)

  fc8 = fc_layer(drop7, "fc8")
  prob = tf.nn.softmax(fc8, name="prob")

  m["prob"] = prob

  return m

def show_caffe_net_input():
  x = net_caffe.blobs['data'].data[0]
  assert x.shape == (3, 224, 224)
  i = deprocess(x)
  skimage.io.imshow(i)
  skimage.io.show()

def same_tensor(a, b):
  return np.linalg.norm(a - b) < 1

def main():
  global tf_activations

  cat = load_image("cat.jpg")
  print "tensorflow session"

  images = tf.placeholder("float", [None, 224, 224, 3], name="images")
  m = inference(images)

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    assert cat.shape == (224, 224, 3)
    batch = cat.reshape((1, 224, 224, 3))
    assert batch.shape == (1, 224, 224, 3)

    out = sess.run([m['prob'], m['relu1_1'], m['conv1_1_weights'], \
        m['conv1_1a'], m['pool5'], m['fc6a']], feed_dict={ images: batch })
    tf_activations = {
      'prob': out[0][0],
      'relu1_1': out[1][0],
      'conv1_1_weights': out[2],
      'conv1_1a': out[3][0],
      'pool5': out[4][0],
      'fc6a': out[5][0],
    }

  top1 = print_prob(tf_activations['prob'])
    ##assert top1 == "n02123045 tabby, tabby cat"
  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  print "graph_def byte size", graph_def.ByteSize()
  graph_def_s = graph_def.SerializeToString()

  save_path = "vgg16.tfmodel"

  with open(save_path, "wb") as f:
    f.write(graph_def_s)

  ipdb.set_trace()
  print "saved model to %s" % save_path


if __name__ == "__main__":
  main()
