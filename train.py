import cv2
import os
import sys
import time
import random
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from matplotlib import pyplot as plt
from PIL import Image
from scipy import misc

import network


def read_rgb_file_names(path):
    file_names = []
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        else:
            if line[-1] == '\n':
                line = line[:-1]
            file_names.append(line.split(',')[0])
    file.close()
    return file_names


def loss_function(pred, gt_depth):
    gt_depth = gt_depth[:, :, :, 0]
    # ignore invalid pixel
    mask = tf.cast(tf.not_equal(gt_depth, tf.constant(0, dtype=tf.float32)), dtype=tf.float32)

    L1_distance = tf.abs(pred[:, :, :, 0] * mask - gt_depth * mask)
    L_depth = tf.reduce_mean(L1_distance)
    gt_confidence = tf.exp(-tf.abs(L1_distance))
    L_confidence = tf.reduce_mean(tf.abs(pred[:, :, :, 1] * mask - gt_confidence * mask))
    L_regular = tf.reduce_mean(tf.abs(pred[:, :, :, 1] * mask))

    # Loss= 0.1*L_d + 0.5*(L_c+0.5*L_r)
    L_final = tf.constant(1.0, dtype=tf.float32) * L_depth + tf.constant(0.5, dtype=tf.float32) * (
            L_confidence + tf.constant(0.5, dtype=tf.float32) * L_regular)
    return L_final


def train(filelist_path, pretrain_model_path, output_models_dir):
    # hyper-params
    initial_lr = 0.01  # initial learning rate
    lr_decay = 0.25  # learning rate decay
    lr_dacay_epoch = 10  # learning rate drops every 10 epochs
    momentum = 0.9  # momentum
    max_iters = 250000  # max itarations
    batch_size = 16  # batch size
    display_step = 20  # display step
    model_save_step = 5000  # model saving step

    # input and output size
    height = 228
    width = 304
    depth_height = 228
    depth_width = 304
    channels_rgb = 3
    channels_depth = 1
    float_to_int_scale = 5000.0

    rgb_filelist = read_rgb_file_names(filelist_path)
    depth_filelist = []
    iter_per_epoch = len(filelist_path) / batch_size
    lr_dacay_iter = lr_dacay_epoch * iter_per_epoch

    # Create a placeholder for the input image and label
    input_node_rgb = tf.placeholder(tf.float32, [batch_size, height, width, channels_rgb])
    input_node_depth = tf.placeholder(tf.float32, [batch_size, depth_height, depth_width, channels_depth])

    # Construct the network
    net = network.DeNet({'data': input_node_rgb}, batch_size, 1.0, 1)
    pred = net.get_output()

    # Loss, Learning rate, optimizer
    loss = loss_function(pred, input_node_depth)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_lr, global_step, lr_dacay_iter, lr_decay, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_locking=False,
                                           name='Momentum', use_nesterov=False).minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # load the pre-train model
        if os.path.exists(pretrain_model_path):
            net.load(pretrain_model_path, sess, True)

        step = 0
        filelist_size = len(rgb_filelist)
        train_ptr = 0
        batch_of_rgb_path = []
        batch_of_depth_path = []
        labels = []

        loss_batch_average = 0

        while step < max_iters:

            # shuffle the filelist when one epoch is finished
            if step % iteration_per_epoch == 0:
                random.shuffle(rgb_filelist)
                depth_filelist = []
                for rgb_path in rgb_filelist:
                    depth_filelist.append(rgb_path.replace("rgb", "depth"))

            # Get next batch of image and depth labels
            if (train_ptr + batch_size) < filelist_size:
                batch_of_rgb_path = rgb_filelist[train_ptr:(train_ptr + batch_size)]
                batch_of_depth_path = depth_filelist[train_ptr:(train_ptr + batch_size)]
                train_ptr += batch_size
            else:
                new_ptr = (train_ptr + batch_size) % filelist_size
                batch_of_rgb_path = rgb_filelist[train_ptr:] + rgb_filelist[:new_ptr]
                batch_of_depth_path = depth_filelist[train_ptr:] + depth_filelist[:new_ptr]
                train_ptr = new_ptr

            # Container for input rgb and depth label
            rgb_images = np.ndarray([batch_size, height, width, channels_rgb])
            depth_images = np.ndarray([batch_size, depth_height, depth_width, channels_depth])

            # read rgb image, resize and crop
            for i, rgb_path in enumerate(batch_of_rgb_path):
                # print rgb_path
                rgb_img = cv2.imread(rgb_path)
                rgb_img = cv2.resize(rgb_img, (320, 240))
                rgb_img = rgb_img[6:234, 8:312]
                rgb_images[i] = rgb_img

            # read depth image, resize and crop
            for j, depth_path in enumerate(batch_of_depth_path):
                # print depth_path
                depth_image = cv2.imread(depth_path, -1)
                depth_image = np.array(depth_image).astype('float32')
                depth_image = depth_image / float_to_int_scale
                depth_image = cv2.resize(depth_image, (320, 240))
                depth_image = depth_image[6:234, 8:312]
                depth_image = np.expand_dims(depth_image, axis=2)
                depth_images[j] = depth_image

            batch_loss, _, _ = sess.run([loss, learning_rate, optimizer],
                                        feed_dict={input_node_rgb: rgb_images, input_node_depth: depth_images})
            loss_batch_average += batch_loss

            # print loss
            if (step + 1) % display_step == 0:
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}  ".format(datetime.now(), step,
                                                                                   loss_batch_average / display_step)
                loss_batch_average = 0

            # save the model
            if step % model_save_step == 0:
                saver = tf.train.Saver()
                saver.save(sess, output_models_dir + '/model%08d' % step)

            step += 1

        print "Training denet finished!"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
       	Train the network.
        ''')
    parser.add_argument('--filelist_path', help='file list path(.txt)', default='')
    parser.add_argument('--pretrain_model_path', help='pretrain model path(.npy). If there is not need, ignore it.',
                        default='')
    parser.add_argument('--output_models_dir', help='directory for saving models', default='')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train(args.filelist_path, args.pretrain_model_path, args.output_models_dir)
    os._exit(0)
