#coding:utf-8
import tensorflow as tf
import lenet5
import os
import utils
import numpy as np
import tfrecords_generateds
import vgg16
import time

BATCH_SIZE = 100
LEARNING_RATE_BASE =  0.005 
LEARNING_RATE_DECAY = 0.99 
REGULARIZER = 0.0001 
STEPS = 48000
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH="./model/" 
MODEL_NAME="lenet_model" 
train_num_examples = 24000

def backward():
    vgg = vgg16.Vgg16()
    x = tf.placeholder(tf.float32,[ BATCH_SIZE,
                                    vgg16.IMAGE_SIZE,
                                    vgg16.IMAGE_SIZE,
                                    vgg16.NUM_CHANNELS]) 
    y_ = tf.placeholder(tf.float32, [None, vgg16.OUTPUT_NODE])
    y = vgg.forward(x, True, REGULARIZER) 
    global_step = tf.Variable(0, trainable=False) 

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) 
    loss = cem + tf.add_n(tf.get_collection('losses')) 

    learning_rate = tf.train.exponential_decay( 
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True) 
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): 
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver() 
    img_batch, label_batch = tfrecords_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess: 
        init_op = tf.global_variables_initializer() 
        sess.run(init_op) 

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) 

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch]) 
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,
                                        vgg16.IMAGE_SIZE,
                                        vgg16.IMAGE_SIZE,
                                        vgg16.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}) 
            if i % 20 == 0: 
                # print(ys)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                if i % 480 == 0:
                	saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)

def main():
    backward()

if __name__ == '__main__':
    main()


