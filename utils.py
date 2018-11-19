#!/usr/bin/python
#coding:utf-8
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl
import cv2

mpl.rcParams['font.sans-serif']=['SimHei'] # 正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False # 正常显示正负号

def load_image(path):
    img = cv2.imread(path) 
    
    # 取出切分后的中心图
    short_edge = min(img.shape[:2]) 
    y = (img.shape[0] - short_edge) // 2  #//取整
    x = (img.shape[1] - short_edge) // 2 
    crop_img = img[y:y+short_edge, x:x+short_edge] 
    
    re_img = cv2.resize(crop_img, (224, 224))

    return re_img

def percent(value):
    return '%.2f%%' % (value * 100)

