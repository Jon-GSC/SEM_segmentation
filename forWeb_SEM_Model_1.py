'''---------------------------------------------------------------------------------------------------------------------
# language:python 3.6
# coding: utf-8
# clean up code and comments for web release on 20190528.

OBJECTIVE
    This code is used for creating a model with U-net architecture.

ACKNOWLEDGEMENTS
    Appreciate github, kaggle, google, csdn, authors, etc for sharing the open source code and library.

INTEREST CONFLICT
    None

---------------------------------------------------------------------------------------------------------------------'''
#-----------------------------------------------------------------------------------------------------------------------
# load libraries
import os
import numpy as np
import pandas as pd
import cv2
import time

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.engine.training import Model

from itools import forWeb_u_net as un
#-----------------------------------------------------------------------------------------------------------------------

v0 = 1    # version

img_size_ori = 256   # original image size
img_size_target = 256  # target image size, in case the variable image size

threshold = {1: [1, 110], 2: [111, 160], 3: [161, 199], 4: [200, 255]}   # threshold for classification

epochs = 15   # iteration
batch_size = 16  # batch size

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def main():
    t_start = time.time()

    current_path = os.path.join(os.path.dirname(__file__), 'All_Images')
    basic_name = f'SEM_model_{v0}'
    save_model_name = basic_name + '.model'

    # Load training/testing images
    train_df = pd.DataFrame(columns = [])

    filelist = []
    for file in sorted(os.listdir(current_path+'/TrainData/Images/')):
        if file.endswith('.tif'):
            filelist.append(os.path.splitext(file)[0])
    train_df['img_name'] = filelist

    filelist = []
    for file in sorted(os.listdir(current_path+'/TrainData/Masks/')):
        if file.endswith('.tif'):
            filelist.append(os.path.splitext(file)[0])
    train_df['masks_name'] = filelist

    # Read images
    train_df["images"] = [np.array(un.Img_pixel(threshold,cv2.imread(current_path+'/TrainData/Images/{}.tif'.format(idx), cv2.IMREAD_GRAYSCALE))) / 255 for idx in tqdm_notebook(train_df.img_name)]
    train_df["masks"] = [np.array(load_img(current_path+'/TrainData/Masks/{}.tif'.format(idx), color_mode = "grayscale")) / 255 for idx in tqdm_notebook(train_df.masks_name)]

    # Calculating the grain coverage and coverage classes
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(un.cov_to_class)

    # train/validation split
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test = train_test_split(
        train_df.index.values,
        np.array(train_df.images.map(un.upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        np.array(train_df.masks.map(un.upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1),
        train_df.coverage.values, test_size=0.2, stratify=train_df.coverage_class, random_state=1588)

    model1 = un.UResNet34(input_shape = (1,img_size_target,img_size_target))
    # print(model1.summary())

    # Augmentation
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    # Training1
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.8, patience=2, min_lr=0.00001, verbose=1)

    history = model1.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint,reduce_lr],
                        verbose=2)

    # plot, save convergence and loss.
    un.plot_convergence(v0, current_path, history)

    # Training2
    model1 = load_model(save_model_name,custom_objects={'my_iou_metric': un.my_iou_metric})
    input_x = model1.layers[0].input
    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = optimizers.adam(lr = 0.005)

    model.compile(loss=un.lovasz_loss, optimizer=c, metrics=[un.my_iou_metric_2])

    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2',
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.8, patience=2, min_lr=0.000001, verbose=1)

    history = model.fit(x_train, y_train,
                        validation_data=[x_valid, y_valid],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[ model_checkpoint,reduce_lr,early_stopping],
                        verbose=2)

    pd.DataFrame(history.history).to_csv(f'TrainHistoryDict2_{v0}.csv')  #save history2

    model = load_model(save_model_name,custom_objects={'my_iou_metric_2': un.my_iou_metric_2, 'lovasz_loss': un.lovasz_loss})

    preds_valid = un.predict_result(model,x_valid,img_size_target)

    # choose threshold by validation data
    thresholds_ori = np.linspace(0.3, 0.7, 41)
    thresholds = np.log(thresholds_ori/(1-thresholds_ori))
    ious = np.array([un.iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
    print(ious)

    # find the threshold.
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]
    # threshold_best = 0.28

    # plot validation masks on 20180928,revised by Jon, obtain results by threshold_best
    preds_valid = np.array(preds_valid)
    preds_valid = np.array([np.round(un.downsample(x)> threshold_best) for x in preds_valid]).tolist()
    # print(preds_valid.shape,preds_valid[5])

    # plot predicting results
    un.plot_overlap_img(v0, current_path, ids_valid, train_df, preds_valid)

    # print(x_train.shape,x_valid.shape, x_test.shape,)
    print(f"Training time used: {(time.time()-t_start)/3600} hours")

if __name__ == '__main__':
    main()
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main=main())
