import tensorflow as tf
import numpy as np
import cv2
from math import ceil

conv_feat_shape = [38, 63, 2048]
wd = 0.0005

def conv_layer(inpt, filter_shape, stride, loc, tr_stat, bn_tr_stat, add_l2_stat):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape, loc, tr_stat, add_l2_stat)
    conv = tf.nn.conv2d(inpt, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = tf.Variable(tf.zeros([out_channels]), trainable=bn_tr_stat)
    gamma = weight_variable([out_channels], loc, bn_tr_stat, add_l2_stat)
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    out = tf.nn.relu(batch_norm)
    return out

def residual_block(inpt, output_depth, down_sample, loc, tr_stat, bn_tr_stat, add_l2_stat, projection=False):
    input_depth = inpt.get_shape().as_list()[3]
    if down_sample:
      filter_ = [1,2,2,1]
      inpt = tf.nn.max_pool(inpt, ksize=filter_, strides=filter_, padding='SAME')
    conv1 = conv_layer(inpt, [1, 1, input_depth, output_depth], 1, loc, tr_stat, bn_tr_stat, add_l2_stat)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], 1, loc, tr_stat, bn_tr_stat, add_l2_stat)
    conv3 = conv_layer(conv2, [1, 1, output_depth, output_depth * 4], 1, loc, tr_stat, bn_tr_stat, add_l2_stat)
    if input_depth != (output_depth * 4):
        if projection:
            input_layer = conv_layer(inpt, [1, 1, input_depth, output_depth], 2, loc, tr_stat, bn_tr_stat)
        else:
            input_layer = tf.pad(inpt, [[0,0], [0,0], [0,0], [0, (output_depth * 4) - input_depth]])
    else:
      input_layer = inpt
    res = conv3 + input_layer
    return res

def weight_variable(shape, loc, tr_stat, add_l2_stat):
  initial = tf.truncated_normal(shape, stddev=0.01)
  weight_decay = tf.mul(tf.nn.l2_loss(initial), wd)
  if np.logical_and(add_l2_stat == True, loc == "base"):
      tf.add_to_collection('weight_losses_base', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "trunk"):
      tf.add_to_collection('weight_losses_trunk', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rpn"):
      tf.add_to_collection('weight_losses_rpn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rcnn"):
      tf.add_to_collection('weight_losses_rcnn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "fc"):
      tf.add_to_collection('weight_losses_fc', weight_decay)
  if tr_stat == True:
      return tf.Variable(initial, name = "weights", trainable = True)
  elif tr_stat == False:
      return tf.Variable(initial, name = "weights", trainable = False)

def bias_variable(shape, loc, tr_stat, add_l2_stat):
  initial = tf.constant(0.0, shape=shape)
  weight_decay = tf.mul(tf.nn.l2_loss(initial), wd)
  if np.logical_and(add_l2_stat == True, loc == "base"):
      tf.add_to_collection('weight_losses_base', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "trunk"):
      tf.add_to_collection('weight_losses_trunk', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rpn"):
      tf.add_to_collection('weight_losses_rpn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rcnn"):
      tf.add_to_collection('weight_losses_rcnn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "fc"):
      tf.add_to_collection('weight_losses_fc', weight_decay)
  if tr_stat == True:
      return tf.Variable(initial, name = "biases", trainable = True)
  elif tr_stat == False:
      return tf.Variable(initial, name = "biases", trainable = False)

def weight_variable_bbox(shape, loc, tr_stat, add_l2_stat):
  initial = tf.truncated_normal(shape, stddev=0.001)
  weight_decay = tf.mul(tf.nn.l2_loss(initial), wd)
  if np.logical_and(add_l2_stat == True, loc == "base"):
      tf.add_to_collection('weight_losses_base', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "trunk"):
      tf.add_to_collection('weight_losses_trunk', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rpn"):
      tf.add_to_collection('weight_losses_rpn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "rcnn"):
      tf.add_to_collection('weight_losses_rcnn', weight_decay)
  elif np.logical_and(add_l2_stat == True, loc == "fc"):
      tf.add_to_collection('weight_losses_fc', weight_decay)
  if tr_stat == True:
      return tf.Variable(initial, name = "weights", trainable = True)
  elif tr_stat == False:
      return tf.Variable(initial, name = "weights", trainable = False)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_nopad(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def rpn_accuracy(x, rl):
  x = np.array(x)
  xmax = np.argmax(x, axis = 1)
  rl = np.array(rl)
  s = xmax == rl
  s = s[rl != -1]
  return s.astype(np.float32)

def bbox_counter(ind):
  zeros = (ind == 0).sum(0)
  ones = (ind == 1).sum(0)
  return zeros.astype(np.float32), ones.astype(np.float32)

def process_img(im, l):
  im = np.array(im)
  '''
  imtest = np.reshape(im, [600,1000,3]).astype(np.uint8)
  for i in range(np.array(l).shape[0]):
    cv2.rectangle(imtest, (l[i,0],l[i,1]), (l[i,2],l[i,3]), (0,255,0))
  cv2.imshow("",imtest)
  cv2.waitKey()
  '''
  imx = np.zeros(im.shape, dtype = np.uint8)
  imx[:,:,0] = im[:,:,2] - 102.9801
  imx[:,:,1] = im[:,:,1] - 115.9465
  imx[:,:,2] = im[:,:,0] - 122.7717
  return imx.astype(np.float32)


def debug_bool(x, y):
  l = y[:,0] == 1
  r = x == 1
  res = np.average(np.equal(l, r))
  return res.astype(np.float32)

def cls_unique(x, lb):
  x = np.array(x)
  lb = np.array(lb)
  u = x[:,0] < x[:,1]
  o_1 = u[np.logical_and((lb != -1),(lb != 0))]
  o_2 = np.sum(o_1)
  o_1_ind = np.asarray(np.where(np.logical_and((u == 1),(lb != -1))))
  n = (u == 1)
  s = np.sum(n)
  i = np.asarray(np.nonzero(n)).astype(np.float32)
  return s.astype(np.float32), o_2.astype(np.float32), i.astype(np.float32)

def flip(img, gt, im_info):
  r = np.random.randint(low=0,high=2)
  img = np.array(img).astype(np.uint8)
  gt = np.array(gt)
  if r == 0:
    return img, gt
  elif r == 1:
    f_img = cv2.flip(img,1)
    f_gt = np.zeros(gt.shape)
    f_gt[:,4] = gt[:,4]
    f_gt[:,0] = im_info[1] - gt[:,2]
    f_gt[:,2] = im_info[1] - gt[:,0]
    f_gt[:,1] = gt[:,1]
    f_gt[:,3] = gt[:,3]
    f_gt = f_gt
    return f_img, f_gt.astype(np.int64)

def get_deconv_filter(f_shape):
    width = f_shape[0]
    height = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)

def upscore_layer(bottom, shape, num_classes, name, ksize=4, stride=1):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value
        if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)
            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
        else:
            new_shape = [shape[0], shape[1], shape[2], num_classes]
        output_shape = tf.pack(new_shape)
        f_shape = [ksize, ksize, num_classes, in_features]
        num_input = ksize * ksize * in_features / stride
        stddev = (2 / num_input)**0.5
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')
    return deconv
