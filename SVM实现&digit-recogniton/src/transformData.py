#!/usr/bin/env python3
# -*-coding:utf-8 -*-
import os
from skimage import io
import torchvision.datasets.mnist as mnist
import numpy
 
 
 
 
train_set = (
    mnist.read_image_file('./train-images.idx3-ubyte'),
    mnist.read_label_file('./train-labels.idx1-ubyte')
)
 
test_set = (
    mnist.read_image_file('./t10k-images.idx3-ubyte'),
    mnist.read_label_file('./t10k-labels.idx1-ubyte')
)
 
print("train set:", train_set[0].size())
print("test set:", test_set[0].size())
 
 
def convert_to_img(train=True):
    if(train):
        # f = open(root + 'train.txt', 'w')
        # data_path = './train/'
        #if(not os.path.exists(data_path)):
        #    os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
            # img_path = data_path + str(i) + '.jpg'
            # io.imsave(img_path, img.numpy())
            txt_path = './train2/' + str(label)[7:8] + '_' + str(i) + '.txt'
            arr = img.numpy()
            arr[arr > 0] = 1
            numpy.savetxt(txt_path, img.numpy(), fmt="%d")
            # f.write(img_path + ' ' + str(label) + '\n')
        # f.close()
    else:
        # f = open(root + 'test.txt', 'w')
        # data_path = './test/'
        #if (not os.path.exists(data_path)):
        #    os.makedirs(data_path)
        for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
            # txt_path = './test2/' + str(label)[7:8] + '_' + str(i) + '.txt'
            img_path = './test5/' + str(label)[7:8] + '_' + str(i) + '.jpg'
            arr = img.numpy()
            # arr[arr > 0] = 1
            # numpy.savetxt(txt_path, img.numpy(), fmt="%d")
            io.imsave( img_path, arr )
 
 
# convert_to_img(True)
convert_to_img(False)
