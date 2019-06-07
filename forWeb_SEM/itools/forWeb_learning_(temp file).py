''' Minerial image segmentation using Deep Learning.
revised by Jon on 2018-05-18
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import os
import time
import math
from math import ceil

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import datetime
import io

np.set_printoptions(threshold=np.nan)
tf.logging.set_verbosity(tf.logging.INFO)

class Network:
    def __init__(self, *layers):
        self.layers = [x for x in layers]
        for layer_id, layer in zip(range(len(layers)), layers):
            layer['name'] = '{}'.format(layer_id)

    def create(self, net):
        for layer in self.layers:
            net = self.conv2d_layer(net, layer)
        for layer in reversed(self.layers):
            net = self.conv2d_transpose_layer(net, layer)
        return net

    def conv2d_layer(self, net, layer):
        layer['input_shape'] = net.shape.as_list()
        with tf.variable_scope('conv2d', reuse=False):
            k_size = layer['kernel_size']
            in_channels = layer['input_shape'][3]
            out_channels = layer['output_channels']
            W = tf.get_variable('W{}'.format(layer['name']), shape=(k_size, k_size, in_channels, out_channels))
            b = tf.Variable(tf.zeros([out_channels]))
        stride = layer['stride']
        net = tf.nn.conv2d(net, W, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.leaky_relu(tf.add(tf.contrib.layers.batch_norm(net), b))

    def conv2d_transpose_layer(self, net, layer):
        with tf.variable_scope('conv2d', reuse=True):
            W = tf.get_variable('W{}'.format(layer['name']))
            b = tf.Variable(tf.zeros([W.shape.as_list()[2]]))
        stride = layer['stride']
        net = tf.nn.conv2d_transpose(net, W,
            tf.stack([tf.shape(net)[0], layer['input_shape'][1], layer['input_shape'][2], layer['input_shape'][3]]),
            strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.leaky_relu(tf.add(tf.contrib.layers.batch_norm(net), b))

    def network_description(self):
        description = ''
        for layer in self.layers:
            description += 'C{}(K{},S{},O{}) '.format(layer['name'], layer['kernel_size'], 
                                                      layer['stride'], layer['output_channels'])
        return description

class ImageSegmenter:
    ''' Image segmentation using Tensor Flow.

    Args:
        height (int)
        width (int)
        channels (int)
        result_path (str): path to the trained tensorflow model.
    '''
    def __init__(self, model_dir, width, height, channels=1, batch_size=100):
        self.width = width
        self.height = height
        self.channels = channels
        self.batch_size = batch_size
        #os.makedirs(model_dir, exist_ok=True) 
        self.classifier = tf.estimator.Estimator(model_fn=self._model_fn, model_dir=model_dir)
        self.network = Network(
            {'kernel_size': 5, 'stride': 2, 'output_channels': 32},
            {'kernel_size': 5, 'stride': 1, 'output_channels': 32},
            {'kernel_size': 5, 'stride': 1, 'output_channels': 64},
            {'kernel_size': 5, 'stride': 1, 'output_channels': 64}
        )
        self.description = self.network.network_description()

    @staticmethod
    def __image_load(image_path, img_ext, is_binary=False, channels=1):
        images = []
        for file in sorted(os.listdir(image_path)):  #load image in order form folder by Jon
            if file.endswith(img_ext):
                filename = os.path.join(image_path, file)
                if channels == 1:
                    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                    if is_binary:
                        img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)[1] 
                else:   # channels == 3
                    img = cv2.imread(filename, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        # print(images[0])
        # return
        return images

    @staticmethod
    def __data_split(inputs, targets, predict,divide):
        threshold = int(divide * len(inputs))
        train = (np.stack(inputs[:threshold]), np.stack(targets[:threshold]))
        evaluate = (np.stack(inputs[threshold:]), np.stack(targets[threshold:]))
        predict = np.stack(predict)

        return train, evaluate, predict

    def prepare_data(self, input_path, target_path, input0_path, img_ext="tif", divide=0.9):
        inputs = self.__image_load(input_path, img_ext)
        targets = self.__image_load(target_path, img_ext, is_binary=True)
        predict = self.__image_load(input0_path, img_ext)
        return self.__data_split(inputs, targets, predict, divide)

    def set_default_data(self, train_data, eval_data, predict_data):
        (self.train_x, self.train_y), (self.eval_x, self.eval_y) = train_data, eval_data
        self.predict_x = predict_data

    def _model_fn(self, features, labels, mode):
        net = tf.cast(tf.reshape(features, [-1, self.height, self.width, self.channels]), tf.float32)
        net = tf.map_fn(tf.image.per_image_standardization, net)
        net = tf.stack(net)
        net = self.network.create(net)
        net = tf.sigmoid(net)
        # if 1:# with tf.device('/gpu:0'):  #using gpu?
        if mode == tf.estimator.ModeKeys.TRAIN:
            targets = tf.cast(tf.reshape(labels, [-1, self.height, self.width, self.channels]), tf.float32)
            loss = tf.sqrt(tf.reduce_mean(tf.square(net - targets)))
            train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions={
                'image': tf.round(net)
            }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:   # mode == tf.estimator.ModeKeys.EVAL:
            targets = tf.cast(tf.reshape(labels, [-1, self.height, self.width, self.channels]), tf.float32)
            accuracy = tf.metrics.accuracy(labels=targets, predictions=tf.round(net))
            loss = tf.sqrt(tf.reduce_mean(tf.square(net - targets)))
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={"accuracy": accuracy})

    def train(self, train_x=None, train_y=None, steps=1000):  # steps=1000 better with GPU
        if not (train_x and train_y):
            train_x, train_y = self.train_x, self.train_y
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_x, y=train_y, batch_size=self.batch_size, num_epochs=None, shuffle=True)
        # print(train_x[0],train_y[0])
        # return
        self.classifier.train(input_fn=input_fn, steps=steps)

    def eval(self, eval_x=None, eval_y=None):
        if not (eval_x and eval_y):
            eval_x, eval_y = self.eval_x, self.eval_y
        input_fn = tf.estimator.inputs.numpy_input_fn( x=eval_x, y=eval_y, batch_size=self.batch_size, shuffle=False)
        return self.classifier.evaluate(input_fn=input_fn)

    def predict(self, predict_x=None):
        if not predict_x:
            predict_x = self.eval_x
            # print('check predict().')
        else:
            predict_x = self.predict_x#[0:,0:,0:]

        #     print('check2')
        input_fn = tf.estimator.inputs.numpy_input_fn(x=predict_x, batch_size=self.batch_size, shuffle=False)
        predict_results = self.classifier.predict(input_fn=input_fn)
        return [np.reshape(x['image'], [self.height, self.width]) for x in predict_results]

def plot_results(images, title, file_name, n_results=12, scalar=3):
    fig, axs = plt.subplots(len(images), n_results, figsize=(n_results*scalar, len(images)*scalar))
    fig.suptitle(title, fontsize=20)
    for row in range(len(images)):
        for col in range(max(n_results, len(images))):
            axs[row][col].imshow(images[row][col], cmap='gray')
    plt.savefig(file_name)

def plot_results0(images, title, file_name, n_results=12,scalar=2): #
    # scalar = len(images)
    fig, axs = plt.subplots(len(images), n_results, figsize=(n_results*scalar, len(images)*scalar))
    fig.suptitle(title, fontsize=20)
    for row in range(len(images)):
        for col in range(max(n_results, len(images))):
            axs[row][col].imshow(images[row][col], cmap='gray')
    plt.savefig(file_name)

# def main(argv):
#     root_path = os.path.join('samples', 'testGSC', 'clay32x32')
#     result_path = os.path.join(root_path, 'results', datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
#     plot_path = os.path.join(result_path, 'plot')
#     os.makedirs(plot_path, exist_ok=True)
#     input_path = os.path.join(root_path, 'inputs')
#     target_path = os.path.join(root_path, 'targets')
#
#     segmenter = ImageSegmenter(os.path.join(result_path, 'save'), 32, 32, batch_size=30)
#     segmenter.set_default_data(*segmenter.prepare_data(input_path, target_path))
#     for i in range(50):
#         segmenter.train()
#         eval_results = segmenter.eval()
#         predict_results = segmenter.predict()
#         plot_results([segmenter.eval_x, segmenter.eval_y, predict_results],
#                      title="Accuracy: {}, Network: {}".format(eval_results['accuracy'], segmenter.description),
#                      file_name=os.path.join(plot_path, 'figure{}.jpg'.format(eval_results['global_step'])))
#
# if __name__ == "__main__":
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)
