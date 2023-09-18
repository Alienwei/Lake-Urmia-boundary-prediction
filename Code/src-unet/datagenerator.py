# -*- coding: utf-8 -*-
"""

@author: 59793
"""
from osgeo import gdal_array
from PIL import Image
import os 
import cv2
import matplotlib.pyplot as plt 
import numpy as np
import math
def load_file_lstm(image_dir,volsize,samples,time):
    li=os.listdir(image_dir)
    li.sort()
    os.chdir(image_dir)
    b=[]
    for filename in li:
      #  print(filename)
        #        img = Image.open(filename)
#        grey = img.convert('L')
#        grey.save(filename)
#        f=plt.imread(filename)
#        gaussian_grey=cv2.GaussianBlur(f,(25,25),0,0)
#        cv2.imwrite(filename, gaussian_grey) 
        a= gdal_array.LoadFile(filename)
             #print(a.shape)
        a=a.reshape((1,*volsize,1))
        b.append(a)
#        print(filename)
    for i in range(samples):
        sample = b[i]
        for j in range(time-1):
            sample=np.concatenate((sample,b[i+j+1]),axis=0)
        sample = sample.reshape((1,time,*volsize,1))
        if(i == 0):
            res = sample
        else:
            res = np.concatenate((res,sample),axis = 0)
    #res.append(sample_res)                                                                # print(sample.shape)
    res = res.reshape((1,samples*time,*volsize,1))
    while True:
        yield(res)
        #k += 1
        #k %= samples
        #print(k)
def load_input_arr(image_dir,volsize,time):
    li=os.listdir(image_dir)
    os.chdir(image_dir)
    li.sort()
    for i,filename in zip(range(len(li)),li):
        a = gdal_array.LoadFile(filename)
        a=a.reshape((1,*volsize,1))
        if(i == 0):
            res = a
        else:
            res= np.concatenate((res,a),axis=0)
    res = res.reshape((1,*res.shape)).astype(np.float32)
    #print(res.shape)
    return res
def load_file_lstm_1(image_dir,volsize,samples,time):
    li=os.listdir(image_dir)
    os.chdir(image_dir)
    li.sort()
    b=[]
    for filename in li:
        print(filename)
        a= gdal_array.LoadFile(filename)
        a=a.reshape((1,*volsize,1))
        b.append(a)
    res3 = []
    for i in range(samples):
        sample = b[i+k]
        for j in range(time-1):
            sample=np.concatenate((sample,b[i+j+k+1]),axis=0)
        if(i == 0):
            res = sample
        else:
            res = np.concatenate((res,sample),axis=0)
            #print(res.shape)
        res3.append(res.reshape((1,samples*time,*volsize,1)))
    print(b[-1].shape)
    res3.append(b[-1])

    return res3
def load_file(image_dir,volsize):
    li=os.listdir(image_dir)
    li.sort()
    os.chdir(image_dir)
    b=[]
    for filename in li:
        print(filename)
#        img = Image.open(filename)
#        grey = img.convert('L')
#        grey.save(filename)
#        f=plt.imread(filename)
#        gaussian_grey=cv2.GaussianBlur(f,(25,25),0,0)
#        cv2.imwrite(filename, gaussian_grey) 
        a= gdal_array.LoadFile(filename)
             #print(a.shape)
        a=a.reshape((1,*volsize,1))
        b.append(a)
#        print(filename)
    i = 0
    while i < len(li):
        yield(b[i])
        #print(i)
        i += 1
        i %= len(li)
def hist(filename):
#    li=os.listdir(image_dir)
#    os.chdir(image_dir)
#    for filename in li:
    img=np.array(Image.open(filename).convert('L'))
    plt.figure("lena")
    arr=img.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)  
    plt.show()
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE);
    #Compute histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    #Convert histogram to simple list
    hist = [val[0] for val in hist]; 
    
    #Generate a list of indices
    indices = list(range(0, 256));
    
    #Descending sort-by-key with histogram value as key
    s = [(x,y) for y,x in sorted(zip(hist,indices), reverse=True)]
    
    #Index of highest peak in histogram
    index_of_highest_peak = s[0][0];
    print(index_of_highest_peak)
    
    #Index of second highest peak in histogram
    index_of_second_highest_peak = s[1][0];
    print(index_of_second_highest_peak)
    return index_of_second_highest_peak
def global_linear_transmation(image_dir): #Set the grayscale range to 0~255
    li=os.listdir(image_dir)
    os.chdir(image_dir)
#    img=np.array(Image.open(atlas_file).convert('L'))
#    atlas_peak = hist(atlas_file)
   # print(atlas_peak)
    a = 1
    for filename in li:
        if(a==1):
            atlas_peak = hist(filename)
            print(atlas_peak)
            a+=1
        img=np.array(Image.open(filename).convert('L'))
        cur_peak = hist(filename)
        print(cur_peak)
        difference = atlas_peak - cur_peak
        print(difference)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if(img[i,j]!= 0):
                    img[i,j] += difference
#                if(img[i,j] == 8):
#                    img[i,j] = 0
        outputImg = Image.fromarray(img)
        outputImg = outputImg.convert('L')
        outputImg.save(filename)  

def load_file_binary(image_dir,volsize):
    li=os.listdir(image_dir)
    os.chdir(image_dir)
    b=[]
    for filename in li:
        #img = Image.open(filename)
        #grey = img.convert('L')
        #grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #Grayscale the input image
        img = cv2.imread(filename, 0)
        ret, binary = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #print("threshold value %s"%ret)
        #cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
        #cv2.imshow("binary0", binary)
        #cv2.imwrite(filename,binary)
        plt.imshow(binary, 'gray')
        a= gdal_array.LoadFile(filename)
             #print(a.shape)
        a=a.reshape((1,*volsize,1))
        b.append(a)
    return b
def data_generator(image_dir,volsize):
    li=os.listdir(image_dir)
    os.chdir(image_dir)
    b=[]
    for filename in li:
        newname = filename
        newname = newname.split(".")
        if newname[-1]=="png":
            newname[-1]="jpg"
            img = Image.open(filename)
            grey = img.convert('L')
            newname=str.join(".",newname)
            grey.save(newname)
            f=plt.imread(newname)
            gaussian_grey=cv2.GaussianBlur(f,(25,25),0,0)
            cv2.imwrite(newname, gaussian_grey) 
            a= gdal_array.LoadFile(newname)
            #print(a.shape)
            a=a.reshape((1,*volsize,1))
            b.append(a)
    return b
def load_file1(image_dir,volsize):
    li=os.listdir(image_dir)
    li.sort()
    os.chdir(image_dir)
    b=[]
    for filename in li:
        img = Image.open(filename)
        grey = img.convert('L')
        grey.save(filename)
        a= gdal_array.LoadFile(filename)
        #print(a.shape)
        a=a.reshape((1,*volsize,1))
        b.append(a)
        print(filename)
    return b
def load_file2(image_dir,filename,volsize):
    li=os.listdir(image_dir)
    os.chdir(image_dir)
    img = Image.open(filename)
    grey = img.convert('L')
    grey.save(filename)
    a= gdal_array.LoadFile(filename)
    #print(a.shape)
    a=a.reshape((1,*volsize,1))
    return a
def cvpr_lstm_gen(gen1,gen3,volsize,batch_size=1):
    zeros = np.zeros((batch_size, *volsize, len(volsize)))
    while True:
        X1 = next(gen1)
        train_Y = next(gen3)
        yield ([X1], [train_Y, zeros])
def cvpr2018_gen(gen1,gen2, gen3, volsize, batch_size=1):
    """ generator used for cvpr 2018 model """
    zeros = np.zeros((batch_size, *volsize, len(volsize)))
    while True:
        X1 = next(gen1)
        X2 = next(gen2)
        train_Y = next(gen3)
        yield ([X1,X2], [train_Y, zeros])
        #print(X1, X2,train_Y)
def erode(image_dir,filename):
    #In the corrosion step of preprocessing, one image at a time, rename the image to 1.png
    os.chdir(image_dir)
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#    kernel=np.ones((100,100),np.uint8)
#    kernel_re = []
#    rows, cols = kernel.shape
#    for i in range(rows):
#        result = [0 if math.sqrt((i-3)**2+(j-3)**2) > 3 else 1 for j in range(cols)]
#        kernel_re.append(result)
#    kernel_re = np.array(kernel_re, np.uint8)
    for i in range(50):
        ero=cv2.erode(img,kernel,iterations=i+1)
        newname = filename.split(".")
        newname[0] = newname[0]+str(i)
        newname=str.join(".",newname)
        #ero.save(newname)
        #plt.imshow(ero)
        cv2.imwrite(newname,ero)
def substract(image_dir):
    #In the difference step in preprocessing, just enter the folder name used for corrosion.
    os.chdir(image_dir)
    img1 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)
    res = cv2.imread('1.png',cv2.IMREAD_GRAYSCALE)
    color = 200
    for i in range(50):
        img2 = cv2.imread('1'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)
        swp = cv2.subtract(img1, img2)
        cv2.imwrite('swp'+str(i)+'.png',swp)
        swp = np.array(Image.open('swp'+str(i)+'.png').convert('L'))
        for i in range(swp.shape[0]):
            for j in range(swp.shape[1]):
                if(swp[i,j]!= 0):
                    res[i,j] = color
        img1= img2
        color -= 2
        cv2.imwrite('res.png',res)
    img2 = np.array(Image.open('149.png').convert('L'))
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if(img2[i,j]!= 0):
                res[i,j] = color
    cv2.imwrite('res.png',res)
    #f=plt.imread('res.jpg')
    gaussian_grey=cv2.GaussianBlur(res,(5,5),0,0)
    cv2.imwrite('res_gau.png', gaussian_grey) 
def gaussian(image_dir):
    os.chdir(image_dir)
    f=plt.imread('res.jpg')
    gaussian_grey=cv2.GaussianBlur(f,(5,5),0,0)
    cv2.imwrite('res_gau.jpg', gaussian_grey) 
def gray(image_dir,filename):
    os.chdir(image_dir)
    img = Image.open(filename)
    grey = np.array(img.convert('L'))
    for i in range(grey.shape[0]):    #size为（256，256）
        for j in range(grey.shape[1]):
            print(grey[i,j])
def binary(image_dir):
    #Binarization
    li=os.listdir(image_dir)
    os.chdir(image_dir)
    for filename in li:
#        img = np.array(Image.open(filename).convert('L'))
#        for i in range(img.shape[0]):
#            for j in range(img.shape[1]):
#                if(img[i,j] < 127):
#                    img[i,j] = 0
#                else:
#                    img[i,j] = 255
        img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
        ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        newname = filename.split(".")
        newname[-1] = "png"
        newname=str.join(".",newname)
        cv2.imwrite(newname,thresh1)
        img = cv2.imread(newname,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(filename,img)


def img_resize(image_dir,filename):
    os.chdir(image_dir)
    image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[0], image.shape[1]
    # Set new image resolution frame
    width_new = 320
    height_new = 160
    # Determine the aspect ratio of an image
    img_new = cv2.resize(image, (width_new, height_new))
    cv2.imwrite(filename, img_new)
def move(image_dir,filename):
    os.chdir(image_dir)
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    M = np.float32([[1,0,10],[0,1,0]])
    dst = cv2.warpAffine(img,M,(1280,560))
    cv2.imwrite(filename, dst)
def cut(image_dir,filename):
    os.chdir(image_dir)
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    cutimg = img[0:560,20:580]
    cv2.imwrite("cut.png", cutimg)
if __name__ == "__main__":

    load_file_lstm("path",(160,320),5,6)
