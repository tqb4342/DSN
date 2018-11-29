# -*- coding:utf-8 -*-
"""
@author:TanQingBo
@file:data_preprocess.py
@time:2018/11/2810:40
"""
from SimpleITK import ReadImage, GetArrayFromImage
import SimpleITK as sitk
import numpy as np
import os
import time

path = 'E:/liverdata/nii/nrrd3D/ContainsOnlyLiver/label/upnrrd/'

def read_nrrd(file_name):
    '''
    读取nrrd体数据文件
    :param file_name:nrrd文件路径
    :return:nd-array，(z,y,x)
    '''
    img = ReadImage(file_name)
    return GetArrayFromImage(img)

def writeimg(arr_slice,s):
    filename = os.path.join('E:/liverdata/nii/nrrd3D/',s)
    print(filename)
    out = sitk.GetImageFromArray(arr_slice)
    print(out.GetDimension())
    sitk.WriteImage(out, filename)

def proprecess():
    label_filename = os.listdir(path)
    for filename in label_filename:
        print(filename)
        label = read_nrrd(os.path.join(path,filename))
        label_shape = label.shape
        print(label_shape)
        M,m = np.max(label),np.min(label)
        print('第'+filename+'组数据的最大最小值：')
        print(M)
        print(m)
        label[label<0] = 1
        writeimg(label,filename)
        M, m = np.max(label), np.min(label)
        print('修改后第' + filename + '组数据的最大最小值：')
        print(M)
        print(m)
        # label = label.tolist()
        # for x in range(0,label_shape[0]):
        #     for y in range(0,label_shape[1]):
        #         print(label[x][y])

if __name__ == '__main__':
    proprecess()
