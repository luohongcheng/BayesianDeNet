import os
import numpy as np
import tensorflow as tf
import time
import cv2
import argparse

import network


def predict(rgb_path, model_path, depth_path, confidence_path):
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    float_to_int_scale = 5000.0

    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    net = network.DeNet({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        # read rgb image
        rgb_img = cv2.imread(rgb_path)
        src_height, src_width, src_channel = rgb_img.shape
        rgb_img = cv2.resize(rgb_img, (width, height))

        # network forward
        rgb_images = np.ndarray([batch_size, height, width, channels])
        rgb_images[0] = rgb_img
        pred = sess.run(net.get_output(), feed_dict={input_node: rgb_images})

        # save depth map
        pred_depth = pred[0, :, :, 0]
        pred_depth[np.where(pred_depth < 0)] = 0
        pred_depth = cv2.resize(pred_depth, (src_width, src_height))
        pred_depth = np.array(pred_depth * float_to_int_scale).astype('uint16')
        cv2.imwrite(depth_path, pred_depth)

        # save confidene map
        pred_conf = pred[0, :, :, 1]
        pred_conf[np.where(pred_conf < 0)] = 0
        pred_conf = cv2.resize(pred_conf, (src_width, src_height))
        pred_conf = np.array(pred_conf * float_to_int_scale).astype('uint16')
        cv2.imwrite(confidence_path, pred_conf)


if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(description='''
        predict the depth and confidence map of a RGB image
        ''')
    parser.add_argument('--rgb_path', help='input rgb path', default='')
    parser.add_argument('--model_path', help='input model path(.ckpt)', default='')
    parser.add_argument('--depth_path', help='saved depth map path', default='')
    parser.add_argument('--confidence_path', help='saved confidence map path', default='')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    predict(args.rgb_path, args.model_path, args.depth_path, args.confidence_path)
    os._exit(0)
