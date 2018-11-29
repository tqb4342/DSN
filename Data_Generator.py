# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:Data_Generator.py
@time:2018/11/2514:01
"""

from SimpleITK import ReadImage, GetArrayFromImage
import SimpleITK as sitk
import numpy as np
import os
import time

X_path = []
Y_path = []

path = 'E:/liverdata/nii/nrrd3D/ContainsOnlyLiver'
BLOCK_SIZE = [16, 512, 512]
stride = [8, 127, 127]

def read_nrrd(file_name):
    '''
    读取nrrd体数据文件
    :param file_name:nrrd文件路径
    :return:nd-array，(z,y,x)
    '''
    img = ReadImage(file_name)
    return GetArrayFromImage(img)

def writeimg(arr_slice,i):
    filename = os.path.join('E:/liverdata/nii/nrrd3D/CompleteData/', 'liver-nor'+str(i)+'.nrrd')
    print(filename)
    arr_slice = arr_slice.reshape(BLOCK_SIZE)
    out = sitk.GetImageFromArray(arr_slice)

    print(out.GetDimension())
    sitk.WriteImage(out, filename)

def createGenerator() :
    mylist = range(3)
    for i in mylist:
        yield i*i

class Data:
    def __init__(self, path, block_size, stride):
        self.root_path = path
        self.BLOCK_SIZE = block_size
        # files = os.listdir(self.root_path)
        raw_data_filename = os.listdir(os.path.join(self.root_path,'orig/upnrrd'))
        label_data_filename = os.listdir(os.path.join(self.root_path,'label/upnrrd-0-1process'))
        # raw_data_filename = list(filter(lambda x: x.startswith('orig'), files))
        # label_data_filename = list(filter(lambda x: x.startswith('seg'), files))
        print(label_data_filename[5])
        print(raw_data_filename[5])
        self.data_filename = list(zip(raw_data_filename, label_data_filename))
        self.stride = stride
        self.gen = self.generator(raw_data_filename,label_data_filename)


    def generator(self,raw,label):
        for i in range(len(raw)):
            # label_ = raw.replace('orig', 'seg')
            print(label[i])
            print(raw[i])
            raw_data = read_nrrd(self.root_path + '/orig/upnrrd/' + raw[i])
            print(raw_data.shape)
            label_data = read_nrrd(self.root_path + '/label/upnrrd-0-1process/' + label[i])
            raw_data = raw_data.reshape([1] + list(raw_data.shape) + [1])
            # print(raw_data.shape)
            label_data = 1.0 * (label_data.reshape([1] + list(label_data.shape)) > 0)
            print(raw_data.shape)
            for z in range(0, raw_data.shape[1] - self.BLOCK_SIZE[0] - 1, self.stride[0]):
                # for y in range(0, raw_data.shape[2] - self.BLOCK_SIZE[1] - 1, self.stride[1]):
                #     for x in range(0, raw_data.shape[3] - self.BLOCK_SIZE[2] - 1, self.stride[2]):
                R = raw_data[:, z:z + self.BLOCK_SIZE[0], 0:0 + self.BLOCK_SIZE[1], 0:0 + self.BLOCK_SIZE[2], :]
                L = label_data[:, z:z + self.BLOCK_SIZE[0], 0:0 + self.BLOCK_SIZE[1], 0:0 + self.BLOCK_SIZE[2]]
                # writeimg(R,n)
                print("fdtsdaf")
                yield R, L


    def next(self):
        print(self.gen)
        return self.gen.__next__()


if __name__ == '__main__':
    data = Data(path, BLOCK_SIZE,stride)
    # data.generator()
    # for z in range(0,-1,1):
    #     # for x in range(0,1,1):
    #     print(z)
    # data = [[1,2,3,5],[[1,2,3,2,4,2,3,5,2,3,5,6],[8,3,4,5,3,2,4,8,8,0,5,9,0,5]],
    #          [[2,5,6,2,6,3,7,8,2,6,5,8],[3,7,5,7,8,3,5,6,8,2,3,7,8,4]],
    #          [[3,4,6,4,7,8,9,1,5,4,7,8],[9,3,5,6,3,8,1,9,4,7,3,1,5,5]]]
    # print(data[:,1:1+1])
    # for i in range(0,255,127):
    #     print(i)
    iteration = 0
    while iteration < 10:
        x,y = data.next()
        # writeimg(y, iteration)
        iteration = iteration + 1
    pass