"""
Code for loading the VGG face model and computing the identity preserving loss.

This code is inherited from the official implementation of ExprGan (https://arxiv.org/abs/1709.03842, https://github.com/HuiDingUMD/ExprGAN).
"""
import numpy as np
import tensorflow as tf

def face_embedding(vgg_weights, images):
    """
    Computes the activations for the layers conv1_2, conv2_2, conv3_2, conv4_2, conv5_2
    of the VGG face model with the vgg_weights with images as input.
    
    @param vgg_weights:pre-trained weights of VGG model loaded from a mat file
    
    @param images: tensor of size [batch_size, 96, 96, 3]
    """
    images = (images+1)/2 * 255
    net = vgg_face(vgg_weights, images)
    return net['conv1_2'], net['conv2_2'], net['conv3_2'], net['conv4_2'], net['conv5_2']

    
def vgg_face(data, input_maps):
    """
    Creates VGG model for face identification from given weights.
    
    @param data: pre-trained weights of VGG model loaded from a mat file
    
    @return: VGG network with data weights
    """
    # read meta info
    meta = data['meta']
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0]
    normalization = meta['normalization']
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
    input_maps = tf.image.resize_images(input_maps, size=[image_size[0], image_size[1]])

    # read layer info
    layers = data['layers']
    current = input_maps - np.array([129.1863, 104.7624, 93.5940]).reshape((1, 1, 1, 3))
    network = {}
    for layer in layers[0]:
        name = layer[0]['name'][0][0]

        if name == 'relu6' or name == 'fc6' or name == 'fc7' or name == 'relu7' or name == 'fc8' or name == 'prob' or name == 'pool5':
            continue
        else:
            layer_type = layer[0]['type'][0][0]
            if layer_type == 'conv':
                if name[:2] == 'fc':
                    padding = 'VALID'
                else:
                    padding = 'SAME'
                stride = layer[0]['stride'][0][0]
                kernel, bias = layer[0]['weights'][0][0]
                bias = np.squeeze(bias).reshape(-1)
                conv = tf.nn.conv2d(current, tf.constant(kernel),
                                    strides=(1, stride[0], stride[0], 1), padding=padding)
                current = tf.nn.bias_add(conv, bias)
            elif layer_type == 'relu':
                current = tf.nn.relu(current)
            elif layer_type == 'pool':
                stride = layer[0]['stride'][0][0]
                pool = layer[0]['pool'][0][0]
                current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                         strides=(1, stride[0], stride[0], 1), padding='SAME')
            elif layer_type == 'softmax':
                current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))

        network[name] = current
    return network

