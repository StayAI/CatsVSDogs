#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import utils
from skimage import io, transform
import matplotlib.pyplot as plt

image_train_path = './data/train'
label_train_path = './data/train'
tfRecord_train = './data/train_cat_dog.tfrecords'

image_test_path = './data/test'
label_test_path = './data/test'
tfRecord_test = './data/test_cat_dog.tfrecords'

image_vali_path = './data/validate'
label_vali_path = './data/validate'
tfRecord_vali = './data/vali_cat_dog.tfrecords'

data_path = './data'
resize_height = 224
resize_width = 224

def write_tfRecord(tfRecordName, image_path, label_path, isTrain=True):
    writer = tf.python_io.TFRecordWriter(tfRecordName)  
    num_pic = 0
    if isTrain:
        start, end = 0, 24000
    else:
        start, end = 24000, 25000
    for i in range(start, end):
        if i % 2 == 0:
            ima_path = image_path + '/' + 'cat.' + str(int(i/2)) + '.jpg'
            labels = [1, 0]
        elif i % 2 == 1:
            ima_path = image_path + '/' + 'dog.' + str(int((i+1)/2)-1) + '.jpg'
            labels = [0, 1]
        img = utils.load_image(ima_path)
        # print(np.array(img).shape)
        img_raw = img.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                })) 
        writer.write(example.SerializeToString())
        num_pic += 1 
        print ("the number of picture:", num_pic)

    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    isExists = os.path.exists(data_path) 
    if not isExists: 
        os.makedirs(data_path)
        print ('The directory was created successfully')
    else:
        print ('directory already exists' )
    write_tfRecord(tfRecord_train, image_train_path, label_train_path, True)
    write_tfRecord(tfRecord_vali, image_vali_path, label_vali_path, False)

def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label': tf.FixedLenFeature([2], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([224 * 224 * 3])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label 
      
def get_tfrecord(num, isTrain=True):
    if isTrain:
        img, label = read_tfRecord(tfRecord_train)
    else:
        img, label = read_tfRecord(tfRecord_vali)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 128,
                                                    capacity = 10000,
                                                    min_after_dequeue = 7000)
    return img_batch, label_batch

def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
