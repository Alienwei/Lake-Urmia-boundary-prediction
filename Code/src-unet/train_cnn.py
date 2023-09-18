"""
train atlas-based alignment with CVPR2018 version of VoxelMorph 
"""

# python imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model 

# project imports
import networks_2d
import losses
import datagenerator
sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen

from osgeo import gdal_array
from PIL import Image
def train(model='vm2',
          model_dir='path/test',
          data_loss='mse',
          batch_size=1,
          initial_epoch=0):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """

    # load atlas from provided files. The atlas we used is 160x192x224.
    vol_size = (640,560)
    # prepare data files
    # for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
    # inside each folder is a /vols/ and a /asegs/ folder with the volumes
    # and segmentations. All of our papers use npz formated data.
    

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32,32,32]
    if model == 'vm1':
        nf_dec = [32, 32, 32, 32, 8, 8]
    elif model == 'vm2':
        nf_dec = [ 32, 32, 32, 32]
    else: # 'vm2double': 
        nf_enc = [f*2 for f in nf_enc]
        nf_dec = [f*2 for f in [32, 32, 32, 32, 32, 16, 16]]

    assert data_loss in ['mse', 'cc', 'ncc'], 'Loss should be one of mse or cc, found %s' % data_loss
    if data_loss in ['ncc', 'cc']:
        data_loss = losses.NCC().loss  
        
    gpu = '/gpu:%d' % 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    config = tf.ConfigProto(allow_soft_placement=True) 
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    with tf.device(gpu):
        model = networks_2d.unet_core(vol_size, nf_enc, nf_dec)
        model.save(os.path.join(model_dir, '%02d.h5' % initial_epoch))
        save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
        save_callback = ModelCheckpoint(save_file_name,period=100)
        mg_model = model
        #mg_model = multi_gpu_model(model,gpus=2)
        mg_model.compile(optimizer=Adam(lr=1e-4), 
                         loss=[data_loss, losses.Grad('l1').loss],
                         loss_weights=[1,0.01],
                         metrics=['acc'])
        X1_gen = datagenerator.load_file('/path/Urmia/1',vol_size)
        X2_gen = datagenerator.load_file('/path/Urmia//2',vol_size)
        train_Y_gen = datagenerator.load_file('/path/Urmia//3',vol_size)
        predict_gen = datagenerator.cvpr2018_gen(X1_gen,X2_gen,train_Y_gen,vol_size)
        # fit
        mg_model.fit_generator(predict_gen,
                     initial_epoch=initial_epoch,
                     steps_per_epoch=1,
                     callbacks=[save_callback],
                     epochs=10000,
                     verbose=1)
  


if __name__ == "__main__":
   

# Source image
     #Convert to grayscale image
    train(model='vm2',model_dir='/path/Urmia/test')
