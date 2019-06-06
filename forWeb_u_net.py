# libraries for training model and plotting...
# some functions are downloaded online with minor revision with acknowledgement to original authors, on 20180818.
#
# reference websites: http://www.github.com;  http://www.kaggle.com;  http://blog.csdn.net
#
# NOTE: some functions are not called in main function, for testing only, ignore those.
#
#
import os
import numpy as np
import pandas as pd
import six
import cv2

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
from skimage.transform import resize

from keras import backend as K
from keras import optimizers
from keras.regularizers import l2
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import Dropout,Add
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.merge import concatenate,add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.losses import binary_crossentropy

import tensorflow as tf

img_size_ori = 256
img_size_target = 256

#-----------------------------------------------------------------------------------------------------------------------
# functions listing:

def Img_pixel(threshold,image):   # freatures by pixel value on 2019-01-12
    if len(np.unique(image))==2 and np.amax(image)==255 and np.amin(image)==0:
        return image
    else:
        threshold0 = {1: [1, 110], 2: [111, 160], 3: [161, 199], 4: [200, 255]}
        # image1 = cv2.filter2D(image, -1, kernel)  # smooth.
        image = cv2.bilateralFilter(image, 5, 75, 75)  # bilateral filtering

        kernel = np.ones((3, 3), np.uint8)

        image_por = cv2.inRange(image, threshold[1][0], threshold[1][1])
        image_por[np.nonzero(image_por)] = threshold0[1][1]
        # image_por = cv2.morphologyEx(image_por,cv2.MORPH_CLOSE,kernel)

        image_org = cv2.inRange(image, threshold[2][0], threshold[2][1])
        image_org[np.nonzero(image_org)] = threshold0[2][1]
        # image_org = cv2.morphologyEx(image_org,cv2.MORPH_CLOSE,kernel)

        image_mat = cv2.inRange(image, threshold[3][0], threshold[3][1])
        image_mat[np.nonzero(image_mat)] = threshold0[3][1]
        # image_mat = cv2.morphologyEx(image_mat,cv2.MORPH_CLOSE,kernel)

        image_pyr = cv2.inRange(image, threshold[4][0], threshold[4][1])
        image_pyr[np.nonzero(image_pyr)] = threshold0[4][1]
        # image_pyr = cv2.morphologyEx(image_pyr,cv2.MORPH_CLOSE,kernel)

        image = image_por + image_org + image_mat + image_pyr

        return image


def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x


def merge(deconv, conv, mode = 'concat', concat_axis = 3):
    uconv = concatenate([deconv, conv])
    return uconv


size = (3, 3)
def build_model(input_layer, start_neurons, dropout=0.5):
    # 256 -> 128
    conv1 = Conv2D(start_neurons * 1, size, activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, size, activation="relu", padding="same")(conv1)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout / 2)(pool1)

    # 128 -> 64
    conv2 = Conv2D(start_neurons * 2, size, activation=None, padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, size, activation=None, padding="same")(conv2)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    # 64 -> 32
    conv3 = Conv2D(start_neurons * 4, size, activation=None, padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, size, activation=None, padding="same")(conv3)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    # 32 -> 16
    conv4 = Conv2D(start_neurons * 8, size, activation=None, padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, size, activation=None, padding="same")(conv4)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, size, activation=None, padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, size, activation=None, padding="same")(convm)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # 16 -> 32
    deconv4 = Conv2DTranspose(start_neurons * 8, size, strides=(2, 2), padding="same")(convm)
    uconv4 = merge(deconv4, conv4, mode = 'concat', concat_axis = 3)
    uconv4 = Dropout(dropout)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, size, activation=None, padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, size, activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # 32 -> 64
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = merge(deconv3, conv3, mode = 'concat', concat_axis = 3)
    uconv3 = Dropout(dropout)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, size, activation=None, padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, size, activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # 64 -> 128
    deconv2 = Conv2DTranspose(start_neurons * 2, size, strides=(2, 2), padding="same")(uconv3)
    uconv2 = merge(deconv2, conv2, mode = 'concat', concat_axis = 3)
    uconv2 = Dropout(dropout)(uconv2)

    uconv2 = Conv2D(start_neurons * 2, size, activation=None, padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, size, activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 128 -> 256
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2) #(5,5)
    uconv1 = merge(deconv1, conv1, mode = 'concat', concat_axis = 3)
    uconv1 = Dropout(dropout)(uconv1)

    uconv1 = Conv2D(start_neurons * 1, size, activation=None, padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, size, activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer


# Get Iou Vector
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.75], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.75], tf.float64)


def my_iou_metric_3(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.75], tf.float64)


# loss functions (src: http://github.com)
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def weighted_bce_loss(y_true, y_pred, weight):
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = weight * (logit_y_pred * (1. - y_true) +
                     K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd
    averaged_mask = K.pool2d(
        y_true, pool_size=(50, 50), strides=(1, 1), padding='same', pool_mode='avg')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight = 5. * K.exp(-5. * K.abs(averaged_mask - 0.5))
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + dice_loss(y_true, y_pred)
    return loss


# code download from: https://github.com/bermanmaxim/LovaszSoftmax,
def lovasz_grad(gt_sorted):
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.elu(errors_sorted) + 1, tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    # logits = K.log(y_pred / (1. - y_pred))
    loss = lovasz_hinge(y_pred, y_true, per_image=True, ignore=None)
    return loss


# https://github.com/raghakot/keras-resnet/blob/master/resnet.py
def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):

        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, block_fn, repetitions, input_tensor):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(img_input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        model = Model(inputs=img_input, outputs=block)
        return model

    @staticmethod
    def build_resnet_34(input_shape, input_tensor):
        return ResnetBuilder.build(input_shape, basic_block, [3, 4, 6, 3], input_tensor)


# U-Net
def UResNet34(input_shape=(256, 256, 1), classes=1, decoder_filters=32, decoder_block_type='upsampling',
              encoder_weights="imagenet", input_tensor=None, activation='sigmoid', **kwargs):
    backbone = ResnetBuilder.build_resnet_34(input_shape=input_shape, input_tensor=input_tensor)

    input_layer = backbone.input  # input = backbone.input
    output_layer = build_model(input_layer, 32, 0.6)  # x
    model = Model(input_layer, output_layer)
    c = optimizers.adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=c,
                  metrics=[my_iou_metric])  # bce_dice_loss  binary_crossentropy
    model.name = 'u-resnet34'
    return model


#used for converting the decoded image to rle mask Fast compared to previous one
def rle_encode(im):

    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# source: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    #temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    # print(temp1)
    intersection = temp1[0]
    # print("temp2 = ",temp1[1])
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    # print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1
        false_positives = np.sum(matches, axis=0) == 0
        false_negatives = np.sum(matches, axis=1) == 0
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'    # '0'display  '3'-no display gpu device running information on 20180823
#-----------------------------------------------------------------------------------------------------------------------
# n_pix = 128,256
#-----------------------------------------------------------------------------------------------------------------------
def img_cut_0(index,n_pix,input_file,datasets_path,n_hw):   # cut image for training dataset folder on 2018-08-15.
    # index=0, cut orginal image, index=1, cut mask

    os.makedirs(datasets_path, exist_ok=True)
    xy_radium = int(n_hw / 2)

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if index == 1:
        image[image != 255] = 0  #cut mask.
    img_size = image.shape

    icount = 0
    y_centre = int(n_hw / 2)
    while y_centre < img_size[0]-xy_radium+1:
        y1 = y_centre - xy_radium
        y2 = y_centre + xy_radium
        x_centre = xy_radium
        while x_centre < img_size[1]-xy_radium+1:
            x1 = x_centre - xy_radium
            x2 = x_centre + xy_radium
            img = image[x1:x2, y1:y2]
            # print(img.shape)

            _, tail0 = os.path.split(input_file)
            filename, ext = os.path.splitext(tail0)
            name = "{}/{}_{:04}.tif".format(datasets_path,filename, icount)
            cv2.imwrite(name, img)   #save

            icount += 1
            x_centre += n_pix
        y_centre += n_pix


def img_cut_1(index,n_pix,input_file,datasets_path,n_hw):   # cut image for dataset folder on 2019-03-29.

    os.makedirs(datasets_path, exist_ok=True)
    xy_radium = int(n_hw / 2)
    xy_crop = int(xy_radium / 2)
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if index ==1:
        image[image != 255] = 0
    img_size0 = image.shape

    image = cv2.copyMakeBorder(image, xy_crop, xy_crop, xy_crop, xy_crop, cv2.BORDER_REFLECT)
    img_size = image.shape

    icount = 0
    y_centre = int(n_hw / 2)
    while y_centre < img_size[0]-xy_radium+1:
        y1 = y_centre - xy_radium
        y2 = y_centre + xy_radium
        x_centre = xy_radium
        while x_centre < img_size[1]-xy_radium+1:
            x1 = x_centre - xy_radium
            x2 = x_centre + xy_radium
            img = image[x1:x2, y1:y2]
            # print(img.shape)
            # exit()

            _, tail0 = os.path.split(input_file)
            filename, ext = os.path.splitext(tail0)
            name = "{}/{}_{:04}.tif".format(datasets_path,filename, icount)
            cv2.imwrite(name, img)   #save

            icount += 1
            x_centre += xy_radium
        y_centre += xy_radium


def merge_images(images):  # merge images
    images = np.array(images)
    n_img = len(images)
    size_img = images[0].shape
    n_img_0 = int(np.sqrt(n_img))
    if n_img_0 != np.floor(n_img_0):
        print('Error!, amount of images is not square!')
    # np.reshape(images,(None,10*128))
    image = images.reshape(n_img_0,n_img_0,size_img[0],size_img[1]).transpose(1,0,2,3).swapaxes(1,2).reshape(size_img[0]*n_img_0,-1)
    return image


def merge_images_1(images):  # merge images on 2018-07-29

    images = np.array(images)
    n_img = len(images)
    images = images[:,64:192,64:192]   #crop

    size_img = images[0].shape
    n_img_0 = int(np.sqrt(n_img))
    if n_img_0 != np.floor(n_img_0):
        print('Error!, amount of images is not square!')
    # np.reshape(images,(None,10*128))
    image = images.reshape(n_img_0,n_img_0,size_img[0],size_img[1]).transpose(1,0,2,3).swapaxes(1,2).reshape(size_img[0]*n_img_0,-1)
    # print(image[0:30,0:30],'\n',type(image),image.shape, n_img, size_img)
    return image


def remove_files(images_path_tmp):
    for root, dirs, files in os.walk(images_path_tmp, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))


def plot_convergence(v0,current_path,history):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,6))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train IOU")
    ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation IOU")
    ax_score.legend()
    plt.savefig(os.path.join(current_path, f'ConvergenceCurve_{v0}.png'))
    pd.DataFrame(history.history).to_csv(f'TrainHistoryDict1_{v0}.csv')   # save history


def plot_overlap_img(v0,current_path,ids_valid,train_df,preds_valid):
    offset = 0
    max_images = 100
    grid_width = 10
    grid_height = int(max_images / grid_width)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    for i, idx in enumerate(ids_valid[offset:offset+max_images]):
        img = train_df.loc[idx].images
        mask = train_df.loc[idx].masks
        pred = preds_valid[i+offset]
        ax = axs[int(i / grid_width), i % grid_width]
        ax.imshow(img, cmap="Greys")
        ax.imshow(mask, alpha=0.3, cmap="Greens")
        ax.imshow(pred, alpha=0.3, cmap="OrRd")
        ax.text(img_size_ori - 1, 1, round(train_df.loc[idx].coverage, 2), color="black", ha="right", va="top")
        ax.text(1, 1, train_df.loc[idx].coverage_class, color="black", ha="left", va="top")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle("Green: Grain, Red: prediction. Top-left: coverage class, top-right: Grain coverage, bottom-left: depth")
    plt.savefig(os.path.join(current_path, f'Evaluation_Result(color)_{v0}.png'))
    plt.show()

