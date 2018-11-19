#coding:utf-8
import csv
import vgg16
import utils
import time
import lenet5
import backward
import numpy as np
import tensorflow as tf
import tfrecords_generateds
import numpy as np

path = './data/test/'

def app():
    csvfile = open("./npy/submission.csv", "w")
    Writer = csv.writer(csvfile,delimiter=',',lineterminator='\r\n')
    Writer.writerow(['id', 'label'])

    with tf.Graph().as_default() as g:

        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
        vgg = vgg16.Vgg16() 
        y = vgg.forward(images, False, None) 
        y = tf.nn.softmax(y)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
                for num in range(1, 5):
                    img_ready = utils.load_image(path + str(num) + '.jpg') / 255.0
                    img_ready = img_ready.reshape((1, 224, 224, 3))

                    probability = sess.run(y, feed_dict={images:img_ready})
                    # print(probability, probability[0][0]+probability[0][1])
                    top5 = np.argsort(probability[0])[-1:-3:-1]
                    
                    row = [num]
                    row.append(probability[0][1])
                    print(num, probability[0][1])
                    Writer.writerow(row)
                csvfile.close()
                print("write finished")

                coord.request_stop()
                coord.join(threads)
            else:
                print('No checkpoint file found')

def main():
    app()

if __name__ == '__main__':
    main()

