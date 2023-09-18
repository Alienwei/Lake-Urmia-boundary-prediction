# -*- coding: utf-8 -*-
"""

@author: 59793
"""
import os
import argparse
import tensorflow as tf
import Networks_STN as Nets
import Params
#import DataHandeling
import numpy as np
import losses
from utils_lstm import log_print
#import requests
import datagenerator
import sys
from osgeo import gdal_array
from PIL import Image
import cv2
try:
    import tensorflow.python.keras as k
except AttributeError:
    import tensorflow.keras as k
# 加载
class AWSError(Exception):
    pass

 
# 预测
def predict(epoch):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #graph = tf.Graph()  
    model_path="path/result"
    model_dir="path/result"
    vol_size=(640,560)
    with tf.device('/gpu:1'):
        data_provider = datagenerator.load_file_lstm_1("path/Urmia/2",vol_size,1,8)
        x = data_provider[0]
        print(x.shape)
        x = tf.cast(x,tf.float32)
        y = data_provider[1]
        print(y.shape)
        y=tf.cast(y,tf.float32)
            
        model = params.net_model(params.net_kernel_params, params.data_format, False)
    
        print_str = 'total_loss is {:f} | flow_loss is {:f} | acc is {:f} | flow_acc is {:f}'
    
        y_pred,flow_pred = model(x, True)
        flow = np.zeros((1, *vol_size, len(vol_size)))
        flow_loss = losses.Grad('l2').loss(flow,flow_pred)
        train_loss = k.metrics.MSE(y,y_pred)
        loss = tf.math.reduce_mean(train_loss)
        total_loss = loss + 0.01 * flow_loss
        acc,acc_op = tf.metrics.accuracy(y, y_pred)
        flow_acc,flow_acc_op = tf.metrics.accuracy(flow, flow_pred)
        saver = tf.train.Saver()
    
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            ckpt = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, ckpt)
            total_loss_val,flow_loss_val,acc_val,acc_op_val,flow_acc_val,flow_op_val,y_val,flow_val = sess.run([total_loss,flow_loss,acc,acc_op,flow_acc,flow_acc_op,y_pred,flow_pred])
            print(print_str.format(total_loss_val,flow_loss_val,acc_op_val,flow_op_val))
#        np.save(os.path.join(model_dir, 'y_pred.npy'), y_val[0,:,:,0])
#        np.save(os.path.join(model_dir, 'flow_pred0.npy'), flow_val[0,:,:,0])
#        np.save(os.path.join(model_dir, 'flow_pred1.npy'), flow_val[0,:,:,1])
            outputImg = Image.fromarray(y_val[0,:,:,0])
            outputImg = outputImg.convert('L')
            outputImg.save(os.path.join(model_dir, 'y_pred'+str(epoch)+'.png'))
        #cv2.imwrite(os.path.join(model_dir,"pred"+str(epoch)+".png"), y_val[0,:,:,0])


            outputImg1 = flow_val[0,:,:,0]
            Img1List = outputImg1.reshape(vol_size[0]*vol_size[1]).tolist()
            A = min(Img1List)
            B = max(Img1List)
            Gxy = (255/(B-A))*abs(outputImg1-A)
            result = np.uint8(Gxy)
            cv2.imwrite(os.path.join(model_dir,"flow_y"+str(epoch)+".png"), result)
            outputImg2 = flow[0,:,:,1]
            Img2List = outputImg2.reshape(vol_size[0]*vol_size[1]).tolist()

            A = min(Img2List)
            B = max(Img2List)
            print(A)
            print(B)
        #Gxy = (255/(B-A))*abs(outputImg2-A)
        #result = np.uint8(Gxy)
        #cv2.imwrite(os.path.join(model_dir,"flow_x"+str(epoch)+".png"), result)
            saver.save(sess=sess, save_path=os.path.join(model_dir, 'test'), global_step=1)
#    with graph.as_default():
#        check_point_file = tf.train.latest_checkpoint(model_path)
#        saver = tf.train.import_meta_graph("{}.meta".format(check_point_file))  
1            
#            
#            flow_loss = losses.Grad('l2').loss(flow,flow_pred)
#            train_loss = k.metrics.MSE(train_y,y_pred)
#            loss = tf.math.reduce_mean(train_loss)
#            total_loss = loss + 0.01 * flow_loss
#            train_accuracy,acc_op = tf.metrics.A(train_y, y_pred)
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    input_args = arg_parser.parse_args() 
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    params = Params.CTCParams(args_dict)
    predict(0)
