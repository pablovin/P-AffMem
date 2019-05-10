"""
Implementations of the layers used in the subnetworks:

- two-dimensional convolution
- two-dimensional transposed convolution
- full connection
- batch normalization

"""
import tensorflow as tf
import numpy as np


def conv2d(input_map, num_filters, size_kernel=5, stride=2, name=None, reuse=False):
    """
    Adds a convolutional layer to the graph.
    
    @param input_map: input tensor
    @param num_filters: number of applied filters (int)
    @param size_kernel: size of the convolution's kernel (int)
    @param stride: size of the convolution's stride (int)
    
    @return: output tensor
    """
    return tf.layers.conv2d(input_map, 
                            filters=num_filters,
                            kernel_size=size_kernel, 
                            strides=stride, 
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            bias_initializer=tf.constant_initializer(0.0),
                            padding="same",
                            reuse=reuse,
                            name=name)

def dense(input_tensor, units, name=None, reuse=False):
    """
    Adds a fully connected layer to the graph.
    
    @param input_map: input tensor
    @param units: number of units of the layer (int)

    @return: output tensor
    """
    return tf.layers.dense(inputs=input_tensor,
                           units=units,
                           use_bias=True,
                           bias_initializer=tf.constant_initializer(0.0),
                           kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                           reuse=reuse,
                           name=name)

def deconv2d(input_map, num_filters, size_kernel=5, stride=2, name=None, reuse=False):
    """
    Adds a transposed convulotional layer to the graph.
    
    @param input_map: input tensor
    @param num_filters: number of applied filters (int)
    @param size_kernel: size of the transposed convolution's kernel (int)
    @param stride: size of the transposed convolution's stride (int)
    
    @return: output tensor
    """
    return tf.layers.conv2d_transpose(input_map, 
                                      filters=num_filters, 
                                      kernel_size=size_kernel, 
                                      strides=stride,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                      bias_initializer=tf.constant_initializer(0.0),
                                      padding="same",
                                      reuse=reuse,
                                      name=name)

def batch_norm(current, name, reuse=False):
    """
    Adds a batch normalization layer to the graph.
    
    @param current: input tensor
    @param name: name of layer (string)
    
    @return: output tensor
    """
    return tf.contrib.layers.batch_norm(current,
                                        scale=False,
                                        scope=name,
                                        reuse=reuse)
                
