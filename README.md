# Lake-Urmia-boundary-prediction
Lake Urmia boundary prediction model based on U-Net


# intro

## environment
1.cuda 9.0
2.Cudnn 7.6.5
3.Tensorflow-gpu 1.12
4.keras 2.2.4(unet-lstm don't need keras)


## Code
1.src-unet, src-unet-lstm folder unet and unet-lstm Model training main code

src-unet:
1.1 train_cnn.py training file (input folders 1, 2, and 3 are model input images 1, 2, and ground truth images respectively)
1.2 datagenerator.py reads data
1.3 networks_2d.py network structure

src-unet-lstm:
train2D trains one sample at a time
train2Ds multiple samples

2.The ext folder is an expansion package and needs to be placed in the same directory as the above folders.

It might be useful to have each folder inside the `ext` folder on your python path. 
assuming voxelmorph is setup at `/path/to/voxelmorph/`:

```
export PYTHONPATH=$PYTHONPATH:/path/to/voxelmorph/ext/neuron/:/path/to/voxelmorph/ext/pynd-lib/:/path/to/voxelmorph/ext/pytools-lib/
```

If you would like to train/test your own model, you will likely need to write some of the data loading code in 'datagenerator.py' for your own datasets and data formats. There are several hard-coded elements related to data preprocessing and format. 

## test
Read the saved model .h5 file and run test_cnn.py


## data
1. Original image: historical sequence image under Google Earth
2. Grayscale: I selected some regularly transformed grayscale images.
3. Preprocessing based on grayscale gradient 2
