# -*- coding: utf-8 -*-
"""

@author: 59793
"""

"""
Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we 
encourage you to explore architectures that fit your needs. 
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""
# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv2D, Conv3D, Activation, Input, UpSampling2D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf
import datagenerator
import os
from osgeo import gdal_array
from PIL import Image
# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.models as nrn_models
import neuron.utils as nrn_utils

# other vm functions
import losses

def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
#   ndims = len(x_in.get_shape()) - 2
#    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

#   Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv2D(nf, kernel_size=(3,3), padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def unet_core(vol_size, enc_nf, dec_nf,  full_size=True):
    """
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = 2
#    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    # inputs
#    if src is None:
#        src = Input(shape=[*vol_size])
#    if tgt is None:
#        tgt = Input(shape=[*vol_size])
    src = Input(shape=[*vol_size,1])
    #src_pred = Input(shape=[512,512,1])
    tgt = Input(shape=[*vol_size,1])
    x_in = concatenate([src, tgt])
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))
    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    #x = np.array([x, x_enc[-2]])
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    #x = np.array([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    #x = np.array([x, x_enc[-4]])
    #x = conv_block(x, dec_nf[3])
    #x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:                             
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        #x = np.array([x, x_enc[0]])
        x = conv_block(x, dec_nf[3])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])
        
    # RNN
  
        #(512,512,16)
    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)  
    flow = Conv(2, kernel_size=(3,3), padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    #(512,512,2)
    # warp the source with the flow
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([tgt, flow])
    #src=(512,512,1) flow=(512,512,2)
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y, flow])
    return model

def p4_predict_model(vol_size, enc_nf, dec_nf):
    module1=unet_core(vol_size,enc_nf,dec_nf)
    X0=Input((*vol_size,1))
    X1=Input((*vol_size,1))
    [X1_pred,flow1]= module1([X0, X1])
    module2=unet_core(vol_size, enc_nf, dec_nf)
    X1_pred1=X1_pred
    X2=Input((*vol_size,1))
    [X2_pred,flow2]= module2( [X1_pred1, X2])
    module3=unet_core(vol_size, enc_nf, dec_nf)
    X2_pred1=X2_pred
    Y=Input((*vol_size,1))
    [y,flow]= module3([X2_pred1,Y])
    model = Model(inputs=[X0, X1, X2, Y], outputs=[y,flow])
    return model



def p9_predict_model(vol_size, enc_nf, dec_nf):
    module1=unet_core(vol_size,enc_nf,dec_nf)
    X0=Input((*vol_size,1))
    X1=Input((*vol_size,1))
    [X1_pred,flow1]= module1([X0, X1])
    module2=unet_core(vol_size, enc_nf, dec_nf)
    X1_pred1=X1_pred
    X2=Input((*vol_size,1))
    [X2_pred,flow2]= module2( [X1_pred1, X2])
    module3=unet_core(vol_size, enc_nf, dec_nf)
    X2_pred1=X2_pred
    X3=Input((*vol_size,1))
    [X3_pred,flow3]= module3([X2_pred1,X3])
    module4=unet_core(vol_size, enc_nf, dec_nf)
    X3_pred1=X3_pred
    X4=Input((*vol_size,1))
    [X4_pred,flow4]= module4([X3_pred1,X4])
    module5=unet_core(vol_size, enc_nf, dec_nf)
    X4_pred1=X4_pred
    X5=Input((*vol_size,1))
    [X5_pred,flow5]= module5([X4_pred1,X5])
    module6=unet_core(vol_size, enc_nf, dec_nf)
    X5_pred1=X5_pred
    X6=Input((*vol_size,1))
    [X6_pred,flow6]= module6([X5_pred1,X6])
    module7=unet_core(vol_size, enc_nf, dec_nf)
    X6_pred1=X6_pred
    X7=Input((*vol_size,1))
    [X7_pred,flow7]= module7([X6_pred1,X7])
    module8=unet_core(vol_size, enc_nf, dec_nf)
    X7_pred1=X7_pred
    Y=Input((*vol_size,1))
    [y,flow]= module8([X7_pred1,Y])
    model = Model(inputs=[X0, X1, X2, X3, X4, X5, X6, X7, Y], outputs=[y,flow])
    return model


def p7_predict_model(vol_size, enc_nf, dec_nf):
    module1=unet_core(vol_size,enc_nf,dec_nf)
    X0=Input((*vol_size,1))
    X1=Input((*vol_size,1))
    [X1_pred,flow1]= module1([X0, X1])
    module2=unet_core(vol_size, enc_nf, dec_nf)
    X1_pred1=X1_pred
    X2=Input((*vol_size,1))
    [X2_pred,flow2]= module2([X1_pred1, X2])
    module3=unet_core(vol_size, enc_nf, dec_nf)
    X2_pred1=X2_pred
    X3=Input((*vol_size,1))
    [X3_pred,flow3]= module3([X2_pred1,X3])
    module4=unet_core(vol_size, enc_nf, dec_nf)
    X3_pred1=X3_pred
    X4=Input((*vol_size,1))
    [X4_pred,flow4]= module4([X3_pred1,X4])
    module5=unet_core(vol_size, enc_nf, dec_nf)
    X4_pred1=X4_pred
    X5=Input((*vol_size,1))
    [X5_pred,flow5]= module5([X4_pred1,X5])
    module6=unet_core(vol_size, enc_nf, dec_nf)
    X5_pred1=X5_pred
    Y=Input((*vol_size,1))
    [y,flow]= module6([X5_pred1,Y])
    model = Model(inputs=[X0, X1, X2, X3, X4, X5, Y], outputs=[y,flow])
    return model
#if __name__ == "__main__":
#    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#        src = datagenerator.load_file2("path","1.png",(1280,560))
#    
#        src = tf.constant(src,tf.float32)
#        #src = tf.expand_dims(src ,axis=-1,name=None,dim=None)
#        flow = np.ones((1280,560,2))
#        for i in range(1280):
#            for j in range(560):
#                flow[i,j,0]=10
#                flow[i,j,1]=0
#        flow_t = tf.constant(flow, tf.float32)
#        
#        flo = tf.expand_dims(flow_t ,axis=-1,name=None,dim=None)
#        flo = tf.expand_dims(flow_t ,axis=0,name=None,dim=None)
#        
#        print(sess.run(flo))
#        print(flo)
#        print(sess.run(src))
#        print(src)
#        y = nrn_layers.SpatialTransformer(interp_method='linear', indexing='ij')([src, flo])
#        y= y.eval()
#    outputImg = Image.fromarray(y[0,:,:,0])
#    outputImg = outputImg.convert('L')
#    os.chdir("path")
#    outputImg.save("pred.png") 
