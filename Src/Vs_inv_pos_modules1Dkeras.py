#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 02/18/2022

@author: aocai (aocai166@gmail.com) Rice University
"""

# models for generator and discriminator in 1D GANs
from __future__ import division
import tensorflow as tf

gf_default = 32
df_default = 32

#weight_init = tf.compat.v1.keras.initializers.he_uniform(seed=None)
weight_init = tf.compat.v1.keras.initializers.he_normal(seed=None)
# weight_init = tf.keras.initializers.glorot_uniform(seed=None)
# weight_init = tf.keras.initializers.random_uniform(seed=None)
bias_init = tf.compat.v1.keras.initializers.constant(value=0)
disp_dim_default = (17, 4)
vs_dim_default = (99, 1)


def discriminatorP(input_shape=vs_dim_default, df=df_default, name='discriminatorP'):
    with tf.variable_scope(name):
        img = tf.keras.layers.Input(shape=input_shape)
        d1 = tf.keras.layers.Conv1D(df, kernel_size=3, strides=1,
                                    kernel_initializer=weight_init, bias_initializer=bias_init,
                                    padding='same', name='d1_conv')(img)
        d2 = d_layer(d1, df * 2, name='conv_layer1')
        d3 = d_layer(d2, df * 4, name='conv_layer2')
        d4 = d_layer(d3, df * 8, name='conv_layer3')
        dc = tf.keras.layers.Flatten()(d4)
        valid1d = tf.keras.layers.Activation('relu')(dc)
        valid = tf.keras.layers.Dense(1, activation='sigmoid', name='Dense_out')(valid1d)

        return tf.keras.Model(img, valid)


def discriminatorG(input_shape=disp_dim_default, df=df_default, name='discriminatorG'):
    with tf.variable_scope(name):
        img = tf.keras.layers.Input(shape=input_shape)
        d1 = tf.keras.layers.Conv1D(df, kernel_size=3, strides=1,
                                    kernel_initializer=weight_init, bias_initializer=bias_init,
                                    padding='same', name='d1_conv')(img)
        d2 = d_layer(d1, df * 2, stride=1, name='conv_layer1')
        d3 = d_layer(d2, df * 4, stride=1, name='conv_layer2')
        d4 = d_layer(d3, df * 8, stride=1, name='conv_layer3')
        dc = tf.keras.layers.Flatten()(d4)
        valid1d = tf.keras.layers.Activation('relu')(dc)
        valid = tf.keras.layers.Dense(1, activation='sigmoid', name='Dense_out')(valid1d)

        return tf.keras.Model(img, valid)


def d_layer(layer_input, filters, f_size=3, stride=2, name="d_layer"):
    """ Discriminator layers """
    with tf.variable_scope(name):
        d = tf.keras.layers.Activation('relu')(layer_input)
        d = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=stride,
                                   kernel_initializer=weight_init,
                                   bias_initializer=bias_init,
                                   padding='same')(d)
        #d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        #d = instance_norm(d, name="layer_norm")
        # Replace batch norm to layer normalization
        d = tf.keras.layers.LayerNormalization()(d)
        return d


def generator_dcnn_G2P(input_shape=disp_dim_default, gf=gf_default, channels=1, name="generatorG2P"):
    with tf.variable_scope(name):
        img = tf.keras.layers.Input(shape=input_shape)
        d1 = tf.keras.layers.Conv1D(gf, kernel_size=3, strides=1,
                                    kernel_initializer=weight_init, bias_initializer=bias_init,
                                    padding='same')(img)

        # Deep Convolution
        d2 = conv1d(d1, gf * 2, stride=1, name='conv_layer1')
        d3 = conv1d(d2, gf * 4, stride=1, name='conv_layer2')
        d4 = conv1d(d3, gf * 8, stride=1, name='conv_layer3')

        d5 = tf.keras.layers.Conv1D(gf * 8, kernel_size=3, strides=1,
                                    kernel_initializer=weight_init,
                                    bias_initializer=bias_init,
                                    padding='same', activation='relu', name='conv_final')(d4)
        dc = tf.keras.layers.Flatten()(d5)
        d_dense = tf.keras.layers.Dense(vs_dim_default[0], activation='tanh', name='Dense_out')(dc)
        dout = tf.reshape(d_dense, [-1, d_dense.shape[1], channels])

        return tf.keras.Model(img, dout)


def generator_dcnn_P2G(input_shape=vs_dim_default, gf=gf_default, channels=4, name="generatorP2G"):
    with tf.variable_scope(name):
        img = tf.keras.layers.Input(shape=input_shape)
        d_flat = tf.keras.layers.Flatten()(img)
        d_dense = tf.keras.layers.Dense(disp_dim_default[0] * gf * 8, activation='relu', name='Dense_model')(d_flat)
        d0 = tf.reshape(d_dense, [-1, disp_dim_default[0], gf * 8])
        d1 = tf.keras.layers.Conv1D(gf * 8, kernel_size=3, strides=1,
                                    kernel_initializer=weight_init, bias_initializer=bias_init,
                                    padding='same')(d0)

        # Deep Convolution
        d2 = conv1d(d1, gf * 4, stride=1, name='conv_layer1')
        d3 = conv1d(d2, gf * 2, stride=1, name='conv_layer2')
        d4 = conv1d(d3, gf, stride=1, name='conv_layer3')

        dout = tf.keras.layers.Conv1D(channels, kernel_size=3, strides=1,
                                    kernel_initializer=weight_init,
                                    bias_initializer=bias_init,
                                    padding='same', activation='tanh', name='conv_final')(d4)

        return tf.keras.Model(img, dout)


def conv1d(layer_input, filters, f_size=3, stride=2, name='conv1d'):
    """ Layers used during downsampling """
    with tf.variable_scope(name):
        d = tf.keras.layers.Activation('relu')(layer_input)
        d = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=stride,
                                   kernel_initializer=weight_init,
                                   bias_initializer=bias_init,
                                   padding='same')(d)
        d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        return d
    
    
def deconv1d(layer_input, skip_input, filters, f_size=4, name='deconv1d'):
    """ Layers used during upsampling """
    with tf.variable_scope(name):
        u = tf.keras.layers.Activation('relu')(layer_input)
        u = tf.keras.layers.UpSampling1D(size=2)(u)
        u = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=1,
                                   kernel_initializer=weight_init,
                                   bias_initializer=bias_init,
                                   padding='same')(u)
        u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
        u = tf.keras.layers.concatenate([u, skip_input], axis=-1)
        return u
    
    
def deconv1d_crop(layer_input, skip_input, filters, f_size=4, name='deconv1d_crop'):
    """ Layers used during upsampling """
    with tf.variable_scope(name):
        u = tf.keras.layers.Activation('relu')(layer_input)
        u = tf.keras.layers.UpSampling1D(size=2)(u)
        u = tf.keras.layers.Cropping1D(cropping=(0, 1))(u)
        u = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=1,
                                   kernel_initializer=weight_init,
                                   bias_initializer=bias_init,
                                   padding='same')(u)
        u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
        u = tf.keras.layers.concatenate([u, skip_input], axis=-1)
        return u


def deconv1d_final(layer_input, filters, f_size=4, name='deconv1d_final'):
    """ Layers used during final upsampling """
    with tf.variable_scope(name):
        u = tf.keras.layers.Activation('relu')(layer_input)
        u = tf.keras.layers.UpSampling1D(size=2)(u)
        u = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=1,
                                   kernel_initializer=weight_init,
                                   bias_initializer=bias_init,
                                   padding='same')(u)
        u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
        u = tf.keras.layers.Activation('relu')(u)
        return u
    

def generator_resnet(input_shape=disp_dim_default, gf=gf_default, name="generator"):
    
    with tf.variable_scope(name):
        
        img = tf.keras.layers.Input(shape=input_shape)
        
        def residule_blocks(x, filters, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [0, 0]], "REFLECT")
            y = tf.keras.layers.Conv1D(filters, kernel_size=ks, strides=s,
                                       kernel_initializer=weight_init,
                                       bias_initializer=bias_init,
                                       padding='valid', name=name + '_c1')(y)
            #y = instance_norm(y, name=name+'_bn1')
            y = tf.keras.layers.BatchNormalization(momentum=0.9, name=name + '_bn1')(y)
            y = tf.keras.layers.Activation('relu')(y)
            y = tf.pad(y, [[0, 0], [p, p], [0, 0]], "REFLECT")
            y = tf.keras.layers.Conv1D(filters, kernel_size=ks, strides=s,
                                       kernel_initializer=weight_init,
                                       bias_initializer=bias_init,
                                       padding='valid', name=name + '_c2')(y)
            #y = instance_norm(y, name=name+'_bn2')
            y = tf.keras.layers.BatchNormalization(momentum=0.9, name=name+'_bn2')(y)
            return y + x
        
        # Edit from Justin Johnson's model
        # The network with 9 blocks consists of c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-1
        c0 = tf.pad(img, [[0, 0], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.keras.layers.Conv1D(gf, kernel_size=7, strides=1, padding='valid', name="g_conv1")(c0)
        # c1 = instance_norm(c1, name='g_bn1')
        c1 = tf.keras.layers.BatchNormalization(momentum=0.9, name="g_bn1")(c1)
        c1 = tf.keras.layers.Activation('relu')(c1)
        
        c2 = tf.pad(c1, [[0, 0], [1, 1], [0, 0]], "REFLECT")
        c2 = tf.keras.layers.Conv1D(gf*2, kernel_size=3, strides=2, padding='valid', name="g_conv2")(c2)
        c2 = tf.keras.layers.BatchNormalization(momentum=0.9, name="g_bn2")(c2)
        c2 = tf.keras.layers.Activation('relu')(c2)
        
        c3 = tf.pad(c2, [[0, 0], [1, 1], [0, 0]], "REFLECT")
        c3 = tf.keras.layers.Conv1D(gf*4, kernel_size=3, strides=2, padding='valid', name="g_conv2")(c3)
        c3 = tf.keras.layers.BatchNormalization(momentum=0.9, name="g_bn2")(c3)
        c3 = tf.keras.layers.Activation('relu')(c3)
        
        # Define G networks with 9 resnet blocks
        r1 = residule_blocks(c3, gf*4, name='g_res1')
        r2 = residule_blocks(r1, gf*4, name='g_res2')
        r3 = residule_blocks(r2, gf*4, name='g_res3')
        r4 = residule_blocks(r3, gf*4, name='g_res4')
        r5 = residule_blocks(r4, gf*4, name='g_res5')
        r6 = residule_blocks(r5, gf*4, name='g_res6')
        r7 = residule_blocks(r6, gf*4, name='g_res7')
        r8 = residule_blocks(r7, gf*4, name='g_res8')
        r9 = residule_blocks(r8, gf*4, name='g_res9')
        
        d1 = deconv1d_res(r9, gf*2, name='g_deconv1')
        d2 = deconv1d_res(d1, gf, name='g_deconv2')
        #d1 = deconv1d_res_concate(r9, c2, gf*2, name='g_deconv1')
        #d2 = deconv1d_res_concate(d1, c1, gf, name='g_deconv2')
        #d2 = tf.keras.layers.Activation('relu')(d2)
        d2 = tf.pad(d2, [[0, 0], [3, 3], [0, 0]], "REFLECT")
        dout = tf.keras.layers.Conv1D(1, kernel_size=7, strides=1,
                                      kernel_initializer=weight_init,
                                      bias_initializer=bias_init,
                                      padding='valid', activation='tanh', name='conv_final')(d2)
        return tf.keras.Model(img, dout)


def deconv1d_res(layer_input, filters, f_size=3, stride=1, name='deconv1d'):
    """ Layers used during upsampling of resnet """
    with tf.variable_scope(name):
        u = tf.keras.layers.UpSampling1D(size=2)(layer_input)
        u = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=1,
                                   kernel_initializer=weight_init,
                                   bias_init=bias_init,
                                   padding='same', name=name+'_conv')(u)
        #u = instance_norm(u, name=name+'_bn')
        u = tf.keras.layers.BatchNormalization(momemtum=0.9, name=name+'_bn')(u)
        u = tf.keras.layers.Activation('relu')(u)
        return u


def deconv1d_res_concate(layer_input, skip_input, filters, f_size=3, stride=1, name='deconv1d'):
    """ Layers used during upsampling of resnet with concatenate """
    with tf.variable_scope(name):
        u = tf.keras.layers.Activation('relu')(layer_input)
        u = tf.keras.layers.UpSampling1D(size=2)(u)
        u = tf.keras.layers.Conv1D(filters, kernel_size=f_size, strides=1,
                                   kernel_initializer=weight_init,
                                   bias_init=bias_init,
                                   padding='same', name=name+'_conv')(u)
        #u = instance_norm(u, name=name+'_bn')
        u = tf.keras.layers.BatchNormalization(momemtum=0.9, name=name+'_bn')(u)
        u = tf.keras.layers.Concatenate([u, skip_input], axis=-1)
        return u
    

def instance_norm(tensor_in, name="instance_norm"):
    with tf.variable_scope(name):
        depth = tensor_in.get_shape()[2]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(tensor_in, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (tensor_in - mean) * inv
        return scale * normalized + offset


def mae_criterion(input, target):
    return tf.reduce_mean(tf.abs(input - target))


def mae_chi_criterion(input, target, sigma):
    return tf.reduce_mean(tf.divide(tf.abs(input - target), sigma))


def mse_criterion(input, target):
    return tf.reduce_mean((input - target)**2)


def mse_chi_criterion(input, target, sigma):
    return tf.reduce_mean(tf.divide((input - target)**2, sigma**2))


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def wasserstein_loss(input, target):
    return tf.reduce_mean(input * target)