import argparse
import os
import pickle
import time
# noinspection PyPackageRequirements
import tensorflow as tf
import Networks_STN as Nets
import Params
#import DataHandeling
import numpy as np
import losses
from utils_lstm import log_print
from PIL import Image
#import requests
import datagenerator
import sys
import cv2

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen
__author__ = 'arbellea@post.bgu.ac.il'

try:
    # noinspection PyPackageRequirements
    import tensorflow.python.keras as k
except AttributeError:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import tensorflow.keras as k

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print(f'Using Tensorflow version {tf.__version__}')
#if not tf.__version__.split('.')[0] == '2':
#    raise ImportError(f'Required tensorflow version 2.x. current version is: {tf.__version__}')


class AWSError(Exception):
    pass


def train():
    vol_size=(640,560)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    #device = '/gpu:1'
    with tf.device('/gpu:0'):
    #with tf.Graph().as_default(),tf.device('/cpu:0'):
    
        config=tf.ConfigProto(allow_soft_placement=True)
        # Data input
        samples=10
        time_step=8
        train_x_set,train_y_set = datagenerator.load_file_lstm_2("/home/thesis/乌尔米湖/1",vol_size,samples,time_step)
        print(len(train_x_set),len(train_y_set))
    
        #iter_num1 = tf.data.Dataset.from_tensor_slices(tf.constant([0,1])).repeat().batch(1)
        #iter1 = iter_num1.make_initializable_iterator()
        #train_x = train_x_set[iter1.get_next()]
        
        
        #iter_num2 = tf.data.Dataset.from_tensor_slices(tf.constant([0,1])).repeat().batch(1)
#        train_y_iter = tf.data.Dataset.from_tensor_slices(train_y_set).repeat().batch(samples) 
        #iter2 = iter_num2.make_initializable_iterator()
        #train_y = train_y_set[iter2.get_next()]
        train_x = tf.placeholder(tf.float32,(1,time_step,*vol_size,1))
        train_y = tf.placeholder(tf.float32,(1,*vol_size,1))
        
            
        
#        with tf.Session() as sess:
#            train_x.eval(session=sess)
#            sess.run([train_x]) 
#            train_y.eval(session=sess)
#            sess.run([train_y]) 
        #print(train_x)
        
        
        #print(train_y.shape)
        #print(train_y.shape)
#        val_data_provider = params.val_data_provider
#        coord = tf.train.Coordinator()
#        train_data_provider.start_queues(coord)
#        val_data_provider.start_queues(coord)

        # Model
        flow = np.zeros((1, *vol_size, len(vol_size)))
        
        model = params.net_model(params.net_kernel_params, params.data_format, False)
        y_pred,flow_pred = model(train_x, True)
        #print(y_pred.shape)
        #print(flow_pred.shape)
        flow_loss = losses.Grad('l2').loss(flow,flow_pred)
        
        train_loss = k.metrics.MSE(train_y,y_pred)
        loss = tf.math.reduce_mean(train_loss)
        total_loss = loss+0.01*flow_loss
        
        train_accuracy,acc_op = tf.metrics.accuracy(train_y, y_pred)
        flow_acc,flow_acc_op = tf.metrics.accuracy(flow, flow_pred)
        # Losses and Metrics

#        ce_loss = losses.WeightedCELoss(params.channel_axis + 1, params.class_weights)
#        seg_measure = losses.seg_measure(params.channel_axis + 1, three_d=False)
        
        #train_seg_measure = k.metrics.Mean(name='train_seg_measure')
        

#        val_loss = k.metrics.Mean(name='val_loss')
#        val_accuracy = k.metrics.SparseCategoricalAccuracy(name='val_accuracy')
#        val_seg_measure = k.metrics.Mean(name='val_seg_measure')

        # Save Checkpoints
        #optimizer = tf.compat.v2.keras.optimizers.Adam(lr=params.learning_rate)
        model_dir="path/Urmia/test"
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01, name='Adam')
        train_op = optimizer.minimize(total_loss)
        total_step=10000
        saver = tf.train.Saver(max_to_keep=50)
       
        print_str = 'spent: {:4} | train_step：{} | total_loss is {:f} | flow_loss is {:f} | acc is {:f} | flow_acc is {:f}'
        with tf.Session(config=config) as sess:   
            #weights=tf.get_variable()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            #for v in tf.global_variables():
                #print(v)
            start_time = time.time()
            #sess.run(train_op)
            for step in range(total_step):
                for i in range(samples):
                    x = train_x_set[i]
                    y = train_y_set[i]
                    print(i)
                    train_val,total_loss_val,train_loss_val,acc,acc_op_val,flow_loss_val,flow_acc_val,flow_acc_op_val,y_val,flow_val = sess.run([train_op,total_loss,train_loss,train_accuracy,acc_op,flow_loss,flow_acc,flow_acc_op,y_pred,flow_pred],feed_dict={train_x:x,train_y:y})
                    print(print_str.format(time.time() - start_time,step, total_loss_val,flow_loss_val,acc,flow_acc_val))
                #print(y_val.shape)
                #print(flow_val.shape)
#            gradients = tape.gradient(train_loss, model.trainable_variables)
#            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            #print(flow_loss_val,total_loss)
                    if step % 100 == 0:
                        outputImg = Image.fromarray(y_val[0,:,:,0])
                        outputImg = outputImg.convert('L')
                        outputImg.save(os.path.join(model_dir, 'y_pred'+str(step+i)+'.png'))
                    
            
                        outputImg1 = flow_val[0,:,:,0]
                        Img1List = outputImg1.reshape(vol_size[0]*vol_size[1]).tolist()
                        A = min(Img1List)
                        B = max(Img1List)
                        Gxy = (255/(B-A))*abs(outputImg1-A)
                        result = np.uint8(Gxy)
                        cv2.imwrite(os.path.join(model_dir,"flow_y"+str(step+i)+".png"), result)
                        outputImg2 = flow[0,:,:,1]
                        Img2List = outputImg2.reshape(vol_size[0]*vol_size[1]).tolist()
            
                        A = min(Img2List)
                        B = max(Img2List)
                        if(A!=B):
                            Gxy = (255/(B-A))*abs(outputImg2-A)
                            result = np.uint8(Gxy)
                            cv2.imwrite(os.path.join(model_dir,"flow_x"+str(step+i)+".png"), result)

                    if(step%10)==0:
                        saver.save(sess=sess, save_path=os.path.join(model_dir, 'train'+str(step)), global_step=step,write_meta_graph=True)

            #ckpt.step.assign_add(1)
            
        #(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, net=model)
#        if params.load_checkpoint:
#
#            if os.path.isdir(params.load_checkpoint_path):
#                latest_checkpoint = tf.train.latest_checkpoint(params.load_checkpoint_path)
#            else:
#                latest_checkpoint = params.load_checkpoint_path
#            try:
#                print(latest_checkpoint)
#                if latest_checkpoint is None or latest_checkpoint == '':
#                    log_print("Initializing from scratch.")
#                else:
#                    ckpt.restore(latest_checkpoint)
#                    log_print("Restored from {}".format(latest_checkpoint))
#
#            except tf.errors.NotFoundError:
#                raise ValueError("Could not load checkpoint: {}".format(latest_checkpoint))
#
#        else:
#            log_print("Initializing from scratch.")
#
#        manager = tf.train.CheckpointManager(ckpt, os.path.join(params.experiment_save_dir, 'tf_ckpts'),
#                                             max_to_keep=params.save_checkpoint_max_to_keep,
#                                             keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)

        #@tf.function
        def train_step(image, label):
            with tf.GradientTape() as tape:
                y_pred, flow = model(train_x, True)
                #loss = ce_loss(label, predictions)
                train_loss = k.metrics.MSE(name='train_loss')+0.01*losses.Grad('l1').loss(flow)
            gradients = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#            ckpt.step.assign_add(1)
#            train_loss(loss)
#            seg_value = seg_measure(label, predictions)
#            if params.channel_axis == 1:
#                predictions = tf.transpose(predictions, (0, 1, 3, 4, 2))
#                label = tf.transpose(label, (0, 1, 3, 4, 2))
            train_accuracy = tf.metrics.accuracy(train_y, y_pred)
            #train_seg_measure(seg_value)
            return y_pred, flow, train_loss

#        @tf.function
#        def val_step(image, label):
#            predictions, softmax = model(image, False)
#            t_loss = ce_loss(label, predictions)
#
#            val_loss(t_loss)
#            seg_value = seg_measure(label, predictions)
#            if params.channel_axis == 1:
#                predictions = tf.transpose(predictions, (0, 1, 3, 4, 2))
#                label = tf.transpose(label, (0, 1, 3, 4, 2))
#            val_accuracy(label, predictions)
#            val_seg_measure(seg_value)
#            return softmax, predictions, t_loss
#
#        train_summary_writer = val_summary_writer = train_scalars_dict = val_scalars_dict = None
#        if not params.dry_run:
#            train_log_dir = os.path.join(params.experiment_log_dir, 'train')
#            val_log_dir = os.path.join(params.experiment_log_dir, 'val')
#            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#
#            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
#            train_scalars_dict = {'Loss': train_loss, 'SEG': train_seg_measure}
#            val_scalars_dict = {'Loss': val_loss, 'SEG': val_seg_measure}



if __name__ == '__main__':

    class AddNets(argparse.Action):
        import Networks_STN as Nets

        def __init__(self, option_strings, dest, **kwargs):
            super(AddNets, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            nets = [getattr(Nets, v) for v in values]
            setattr(namespace, self.dest, nets)


    class AddReader(argparse.Action):

        def __init__(self, option_strings, dest, **kwargs):
            super(AddReader, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            reader = getattr(DataHandeling, values)
            setattr(namespace, self.dest, reader)


    class AddDatasets(argparse.Action):

        def __init__(self, option_strings, dest, *args, **kwargs):

            super(AddDatasets, self).__init__(option_strings, dest, *args, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):

            if len(values) % 2:
                raise ValueError("dataset values should be of length 2*N where N is the number of datasets")
            datastets = []
            for i in range(0, len(values), 2):
                datastets.append((values[i], (values[i + 1])))
            setattr(namespace, self.dest, datastets)


    arg_parser = argparse.ArgumentParser(description='Run Train LSTMUnet Segmentation')
    arg_parser.add_argument('-n', '--experiment_name', dest='experiment_name', type=str,
                            help="Name of experiment")
    arg_parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                            help="Visible GPUs: example, '0,2,3'")
    arg_parser.add_argument('--dry_run', dest='dry_run', action='store_const', const=True,
                            help="Do not write any outputs: for debugging only")
    arg_parser.add_argument('--profile', dest='profile', type=bool,
                            help="Write profiling data to tensorboard. For debugging only")
    arg_parser.add_argument('--root_data_dir', dest='root_data_dir', type=str,
                            help="Root folder containing training data")
    arg_parser.add_argument('--data_provider_class', dest='data_provider_class', type=str, action=AddReader,
                            help="Type of data provider")
    arg_parser.add_argument('--dataset', dest='train_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs: DatasetName, SequenceNumber")
    arg_parser.add_argument('--val_dataset', dest='val_sequence_list', type=str, action=AddDatasets, nargs='+',
                            help="Datasets to run. string of pairs DatasetName, SequenceNumber")
    arg_parser.add_argument('--net_gpus', dest='net_gpus', type=int, nargs='+',
                            help="gpus for each net: example: 0 0 1")
    arg_parser.add_argument('--net_types', dest='net_types', type=int, nargs='+', action=AddNets,
                            help="Type of nets")
    arg_parser.add_argument('--crop_size', dest='crop_size', type=int, nargs=2,
                            help="crop size for y and x dimensions: example: 160 160")
    arg_parser.add_argument('--train_q_capacity', dest='train_q_capacity', type=int,
                            help="Capacity of training queue")
    arg_parser.add_argument('--val_q_capacity', dest='val_q_capacity', type=int,
                            help="Capacity of validation queue")
    arg_parser.add_argument('--num_train_threads', dest='num_train_threads', type=int,
                            help="Number of train data threads")
    arg_parser.add_argument('--num_val_threads', dest='num_val_threads', type=int,
                            help="Number of validation data threads")
    arg_parser.add_argument('--data_format', dest='data_format', type=str, choices=['NCHW', 'NWHC'],
                            help="Data format NCHW or NHWC")
    arg_parser.add_argument('--batch_size', dest='batch_size', type=int,
                            help="Batch size")
    arg_parser.add_argument('--unroll_len', dest='unroll_len', type=int,
                            help="LSTM unroll length")
    arg_parser.add_argument('--num_iterations', dest='num_iterations', type=int,
                            help="Maximum number of training iterations")
    arg_parser.add_argument('--validation_interval', dest='validation_interval', type=int,
                            help="Number of iterations between validation iteration")
    arg_parser.add_argument('--load_checkpoint', dest='load_checkpoint', action='store_const', const=True,
                            help="Load from checkpoint")
    arg_parser.add_argument('--load_checkpoint_path', dest='load_checkpoint_path', type=str,
                            help="path to checkpoint, used only with --load_checkpoint")
    arg_parser.add_argument('--continue_run', dest='continue_run', action='store_const', const=True,
                            help="Continue run in existing directory")
    arg_parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                            help="Learning rate")
    arg_parser.add_argument('--class_weights', dest='class_weights', type=float, nargs=3,
                            help="class weights for background, foreground and edge classes")
    arg_parser.add_argument('--save_checkpoint_dir', dest='save_checkpoint_dir', type=str,
                            help="root directory to save checkpoints")
    arg_parser.add_argument('--save_log_dir', dest='save_log_dir', type=str,
                            help="root directory to save tensorboard outputs")
    arg_parser.add_argument('--tb_sub_folder', dest='tb_sub_folder', type=str,
                            help="sub-folder to save outputs")
    arg_parser.add_argument('--save_checkpoint_iteration', dest='save_checkpoint_iteration', type=int,
                            help="number of iterations between save checkpoint")
    arg_parser.add_argument('--save_checkpoint_max_to_keep', dest='save_checkpoint_max_to_keep', type=int,
                            help="max recent checkpoints to keep")
    arg_parser.add_argument('--save_checkpoint_every_N_hours', dest='save_checkpoint_every_N_hours', type=int,
                            help="keep checkpoint every N hours")
    arg_parser.add_argument('--write_to_tb_interval', dest='write_to_tb_interval', type=int,
                            help="Interval between writes to tensorboard")
    sys_args = sys.argv

    input_args = arg_parser.parse_args()
    args_dict = {key: val for key, val in vars(input_args).items() if not (val is None)}
    print(args_dict)
    params = Params.CTCParams(args_dict)
    # params = Params.CTCParamsNoLSTM(args_dict)

    # try:
    #     train()
    # finally:
    #     log_print('Done')
    train()
