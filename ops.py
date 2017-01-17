import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *
import pdb
class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, batch_size, epsilon=1e-5, momentum = 0.1, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.batch_size = batch_size

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name=name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        with tf.variable_scope(self.name) as scope:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
            self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))

            self.mean, self.variance = tf.nn.moments(x, [0, 1, 2])

            return tf.nn.batch_norm_with_global_normalization(
                x, self.mean, self.variance, self.beta, self.gamma, self.epsilon,
                scale_after_normalization=True)

def binary_cross_entropy_with_logits(logits, targets, name=None):
    """Computes binary cross entropy given `logits`.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
    """
    eps = 1e-12
    with ops.op_scope([logits, targets], name, "bce_loss") as name:
        logits = ops.convert_to_tensor(logits, name="logits")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(logits * tf.log(targets + eps) +
                              (1. - logits) * tf.log(1. - targets + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat(3, [x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,padding='SAME',
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def _bottleneck_block(name,input_,num_unit,output_dim1,output_dim2,train):
    for i in range(0,num_unit):
        ds = (i ==0)
        if i ==0:
            unit_name = '%sa' % name
        else:
            unit_name = '%s%s' % (name,i)
        x= bottleneck_unit(unit_name, input_, output_dim1, output_dim2,train)
    return x

def bottleneck_block_letters(name,input_,num_unit,output_dim1,output_dim2,train=True):
    return _bottleneck_block(name,input_,num_unit,output_dim1,output_dim2,train)


def bottleneck_unit(name,input_,output_dim1,output_dim2,train):
    in_chans = input_.get_shape()[3]
    batch_size  = input_.get_shape()[0]
    b1_bn = batch_norm(batch_size,name='bn%s_branch1' % name)
    b2a_bn = batch_norm(batch_size,name='bn%s_branch2a' % name)
    b2b_bn = batch_norm(batch_size,name='bn%s_branch2b' % name)
    b2c_bn = batch_norm(batch_size,name='bn%s_branch2b' % name)

    with tf.variable_scope('res%s' % name):
        if in_chans == output_dim2:
            b1 = input_
        else:
            with tf.variable_scope('branch1'):
                b1 = conv2d(input_,output_dim2,d_h =1,d_w =1,name='res%s_branch1' % name)
                b1 = b1_bn(b1,train=train)
        with tf.variable_scope('branch2a'):
                b2 = conv2d(input_,output_dim1,d_h =1,d_w =1,name='res%s_branch2a' % name)
                b2 = b2a_bn(b2,train=train)
                b2 = tf.nn.relu(b2)
        with tf.variable_scope('branch2b'):
                b2 = conv2d(b2,output_dim1,d_h =1,d_w =1,name='res%s_branch2b' % name)
                b2 = b2b_bn(b2,train=train)
                b2 = tf.nn.relu(b2)
        with tf.variable_scope('branch2c'):
                b2 = conv2d(b2,output_dim2,d_h =1,d_w =1,name='res%s_branch2c' % name)
                b2 = b2c_bn(b2,train=train)
        input_ = b1+b2 
        return tf.nn.relu(input_)



def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])
        #deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
        #                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def scale_invariant(output,GT,mask,light):
    #num_elt = tf.to_float(GT.get_shape()[0])
    num_elt = tf.to_float(GT.get_shape()[0] *GT.get_shape()[1] * GT.get_shape()[2] *GT.get_shape()[3])
    #computing GT log
    GT = tf.div(tf.add(GT,1.),2.)
    GT = tf.clip_by_value(GT,1e-10,1.0)
    gt_log  = tf.log(GT)
    
    ### reconstruct NIR image ###
    #norm surface normal
    tmp = tf.sqrt(tf.reduce_sum(tf.square(output),3))
    tmp = tf.expand_dims(tmp,-1)
    """
    exp10 = tf.ones_like(tmp)
    exp10  = tf.mul(exp10,1e-10)
    tmp2 = tf.equal(tmp,tf.constant(0.0))
    tmp = tf.select(tmp2,exp10,tmp)
    tmp = tf.expand_dims(tmp,-1)
    """
    output_nor = tf.div(output,tmp)
    recon_NIR = tf.expand_dims(tf.reduce_sum(tf.mul(output_nor,light),3),-1) 
    recon_NIR2 = tf.div(tf.add(recon_NIR,1.0),2.0) # convert to 0~1
    #recon_NIR2 = tf.mul(recon_NIR2,mask)
    #recon_NIR2 = tf.clip_by_value(recon_NIR2,1e-10,1.0)
    recon_NIR_log = tf.log(recon_NIR2)
    #########computing scale invariant ########
    diff_log = tf.sub(recon_NIR,GT)
    #scale_inv1 = tf.div(tf.reduce_sum(tf.square(diff_log)),tf.to_float(GT.get_shape()[0]))
    scale_inv1 = tf.div(tf.reduce_sum(tf.square(diff_log)),num_elt)
    scale_inv2 = tf.square(tf.reduce_sum(diff_log))
    scale_inv3 = tf.div(scale_inv2,tf.square(num_elt))
    scale_inv = tf.abs(scale_inv1 - scale_inv3*0.001)
    return [scale_inv,recon_NIR]


