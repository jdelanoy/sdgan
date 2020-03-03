import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def GeneratorCNN(z, hidden_num, output_num, repeat_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = int(np.prod([8, 8, hidden_num]))
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        
        for idx in range(repeat_num):
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
            if idx < repeat_num - 1:
                x = upscale(x, 2, data_format)

        out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

    variables = tf.contrib.framework.get_variables(vs)
    return out, variables

def DiscriminatorCNN(x, input_channel, zc_num, zi_num, repeat_num, hidden_num, data_format):
    ninstances = 2
    # x is (Gz + x, ninstances * nchannels, 64, 64)
    print x.get_shape()
    Gz_and_x_instances = tf.split(x, ninstances, axis=1)

    variables = []

    reuse = False
    towers = []
    for x in Gz_and_x_instances:
        print x.get_shape()
        with tf.variable_scope("De", reuse=reuse) as vs:
            # Encoder
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                    #x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            x = slim.fully_connected(x, zc_num + zi_num, activation_fn=None)
            print x.get_shape()
            towers.append(x)
        reuse=True

    variables += tf.contrib.framework.get_variables(vs)

    # concat towers
    x = tf.concat(towers, axis=1)
    with tf.variable_scope("Dz") as vs:
        z = slim.fully_connected(x, zc_num + zi_num + zi_num, activation_fn=None)
    print z.get_shape()
    variables += tf.contrib.framework.get_variables(vs)

    # decode (zc + zi + zi) bottleneck into Ninstances (zc + zi)
    Ddzs = []
    for i in xrange(ninstances):
        with tf.variable_scope("Ddz_{}".format(i)) as vs:
            Ddz = slim.fully_connected(z, zc_num + zi_num, activation_fn=None)
            Ddzs.append(Ddz)
            print Ddz.get_shape()
        variables += tf.contrib.framework.get_variables(vs)

    reuse = False
    out_instances = []
    for Ddz in Ddzs:
        with tf.variable_scope("Dd", reuse=reuse) as vs:
            # Decoder
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(Ddz, num_output, activation_fn=None)
            x = reshape(x, 8, 8, hidden_num, data_format)
            
            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = upscale(x, 2, data_format)

            out = slim.conv2d(x, input_channel / ninstances, 3, 1, activation_fn=None, data_format=data_format)
            print out.get_shape()
            out_instances.append(out)
        reuse = True
    variables += tf.contrib.framework.get_variables(vs)
    out = tf.concat(out_instances, axis=1)
    print out.get_shape()

    return out, z, variables

def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x

def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x

def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h*scale, w*scale), data_format)
