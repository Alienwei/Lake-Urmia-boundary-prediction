import os

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
from keras.optimizers import Adam
import cv2
import networks_2d
import datagenerator
from osgeo import gdal_array
from PIL import Image
import sys
sys.path.append('../ext/neuron/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
sys.path.append('../ext/medipy-lib')
import neuron.utils as utils
import matplotlib.pyplot as plt
from medipy.metrics import dice
import losses
def test(model_name,
        epoch,
        vol_size,
        X1,X2,X3,
         compute_type = 'CPU',  # GPU or CPU
         nf_enc=[16,32,32,32], nf_dec=[32,32,32,32]):
    """
    test

    nf_enc and nf_dec
    #nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """  
    

    # Anatomical labels we want to evaluate

    gpu = '/gpu:%d' % 1
    # gpu handling
    os.environ["CUDA_VISIBLE_DEVICES"] ="-1"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device('/cpu:0'):
        net = networks_2d.unet_core(vol_size,nf_enc, nf_dec)
        net.load_weights(model_name)
        #json = net.to_json()
        #net = tf.keras.models.model_from_json(json)
        pred = net.predict([X1,X2])
        result = pred[0]
        flow = pred[1]

        net.compile(optimizer=Adam(lr=1e-4), 
        loss=['mse', losses.Grad('l2').loss],loss_weights=[1,0.01],metrics=['acc','mae'])                
        flow_t = np.zeros((1, *vol_size, len(vol_size)))
        test_acc_mae = net.evaluate([X1,X2],[X3,flow_t])
        print(test_acc_mae)
        #net.compile(optimizer=Adam(lr=1e-4),
        #loss=['mse', losses.Grad('l2').loss],loss_weights=[1,0.01],metrics=['mae'])
        #flow_t = np.zeros((1, *vol_size, len(vol_size)))
        #test_mae = net.evaluate([X1,X2],[X3,flow_t])
        #print(test_mae)

    outputImg = Image.fromarray(result[0,:,:,0])
    outputImg = outputImg.convert('L')
    outputImg.save("path/result/out"+str(epoch)+".png")
    outputImg1 = flow[0,:,:,0]
    Img1List = outputImg1.reshape(vol_size[0]*vol_size[1]).tolist()
    A = min(Img1List)
    B = max(Img1List)
   # print(A)
   # print(B)
    Gxy = (255/(B-A))*abs(outputImg1-A)
    result = np.uint8(Gxy)
    cv2.imwrite("flow_y"+str(epoch)+".png", result)
    outputImg2 = flow[0,:,:,1]
    Img2List = outputImg2.reshape(vol_size[0]*vol_size[1]).tolist()
    A = min(Img2List)
    B = max(Img2List)
    #print(A)
    #print(B)
    Gxy = (255/(B-A))*abs(outputImg2-A)
    result = np.uint8(Gxy)
    cv2.imwrite("flow_x"+str(epoch)+".png", result)
    #outputImg1 = outputImg1.convert('L')
    #outputImg1.save("path/result/flow_0.jpg")
    #outputImg2 = Image.fromarray(flow[0,:,:,1])
    #outputImg2 = outputImg2.convert('L')
    #outputImg2.save("path/result/flow_1.jpg")
    #matplotlib.image.imsave(path\data\out.png',result[0,:,:,0] )

      
if __name__ == "__main__":
    vol_size=(640,560)
    epoch = 10000
    datadir="path/result"
    li=datagenerator.load_file1(datadir,vol_size)
    X1=li[0]
    X2=li[1]
    X3=li[2]
    test("/path/test1/10000.h5",epoch,vol_size,X1,X2,X3)

