#coding:utf-8
import time
import tensorflow as tf
import lenet5
import backward
import numpy as np
import vgg16
import tfrecords_generateds


TEST_INTERVAL_SECS = 5
TEST_NUM = 100

def test():
    with tf.Graph().as_default() as g: 
        x = tf.placeholder(tf.float32,[ TEST_NUM,
                                        vgg16.IMAGE_SIZE,
                                        vgg16.IMAGE_SIZE,
                                        vgg16.NUM_CHANNELS]) 
        y_ = tf.placeholder(tf.float32, [None, 2])
        vgg = vgg16.Vgg16()
        y = vgg.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
		
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

        img_batch, label_batch = tfrecords_generateds.get_tfrecord(TEST_NUM, isTrain=False)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] 
					
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                    score = []
                    for i in range(10):
                        xs, ys = sess.run([img_batch, label_batch])
                        reshaped_xs = np.reshape(xs,(
                                                TEST_NUM,
                                                vgg16.IMAGE_SIZE,
                                                vgg16.IMAGE_SIZE,
                                                vgg16.NUM_CHANNELS))
                        accuracy_score = sess.run(accuracy, feed_dict={x:reshaped_xs,y_:ys})
                        score.append(accuracy_score)

                    print("After %s training step(s), test accuracy = %g" % (global_step, np.mean(score)))
                    # print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    coord.request_stop()
                    coord.join(threads)
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS) 

def main():
    test()

if __name__ == '__main__':
    main()
