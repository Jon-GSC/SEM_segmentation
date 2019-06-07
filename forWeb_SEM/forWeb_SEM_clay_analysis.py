'''=====================================================================================================================
OBJECTIVE
    This code is the architecture for texture feature extraction and object segmentation of SEM core image.

CREDIT
    Co-authorship is appropriate if your paper benefits from use this study model

NOTE
    If you find a bug, please report it to Jon Liu (jon.liu@canada.ca)
    Thanks for your interest!

HISTORY
    load SEM and extract features with encode-decode model by Jon on 2018-03-19
    tested different u-net architectures on 20180825
    tested different threshold for boundary improvement on 201811110
    clean up code and comments for web release on 20190528.

CONTACT
    Jon Liu,
    Email: jon.liu@canada.ca
    Phone: +1 (403)463-2068

====================================================================================================================='''
#
import sys
import os
import gc
import shutil
import time
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm_notebook

from datetime import timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itools import forWeb_image_process as ip
from itools import forWeb_u_net as un

# import LD_LIBRARY_PATH = /usr/local/cuda-9.0/lib64  to Run-edit configurations-
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

threshold = {1:[1, 110],2:[111, 160],3:[161, 199],4:[200, 255]}  # pore/organic/matrix/pyrite

img_size = 256    # image size
threshold_best = 0.28  # boundary parameter

prj_fold = 'P7'  # SEM image folder for segmentation
v0 = 2  # version

#-----------------------------------------------------------------------------------------------------------------------
current_path = os.path.join(os.path.dirname(__file__),'All_Images')
basic_name = f'SEM_model_{v0}'
save_model_name = basic_name + '.model'
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Params and helpers
def Img_pixel(image):   # extracting freatures by pixel value on 2018-03-12
    if len(np.unique(image))==2 and np.amax(image)==255 and np.amin(image)==0:
        return image
    else:
        threshold0 = {1: [1, 110], 2: [111, 160], 3: [161, 199], 4: [200, 255]}
        image = cv2.bilateralFilter(image, 5, 75, 75)  # bilateral filtering

        # kernel = np.ones((3, 3), np.uint8)
        image_por = cv2.inRange(image, threshold[1][0], threshold[1][1])
        image_por[np.nonzero(image_por)] = threshold0[1][1]

        image_org = cv2.inRange(image, threshold[2][0], threshold[2][1])
        image_org[np.nonzero(image_org)] = threshold0[2][1]

        image_mat = cv2.inRange(image, threshold[3][0], threshold[3][1])
        image_mat[np.nonzero(image_mat)] = threshold0[3][1]

        image_pyr = cv2.inRange(image, threshold[4][0], threshold[4][1])
        image_pyr[np.nonzero(image_pyr)] = threshold0[4][1]

        image = image_por + image_org + image_mat + image_pyr

        return image

def prediction_unet(idx, input_file, input_path, output_path, fn_img, threshold):
# Loading of training/testing image and input model
    model = load_model(save_model_name, custom_objects={'my_iou_metric_2': un.my_iou_metric_2, 'lovasz_loss': un.lovasz_loss})
    model.summary()

    images_path_tmp = os.path.join(input_path, 'tmp')
    un.remove_files(images_path_tmp)

    un.img_cut_0(0, img_size, input_file, images_path_tmp, img_size)
    test_df = pd.DataFrame(columns = [])
    filelist = []
    for file in sorted(os.listdir(images_path_tmp)):
        if file.endswith('.tif'):
            filelist.append(os.path.splitext(file)[0])
    test_df['test_name'] = filelist

    test_df["images"] = [np.array(Img_pixel(cv2.imread(images_path_tmp+'/{}.tif'.format(idx), cv2.IMREAD_GRAYSCALE))) / 255.0 for idx in tqdm_notebook(test_df.test_name)]
    x_test= np.array(test_df.images.map(un.upsample).tolist()).reshape(-1, img_size, img_size, 1)

    #testing and plot.
    t1 = time.time()
    preds_test = un.predict_result(model,x_test,img_size)
    preds_test = np.array([np.round(un.downsample(x) > threshold_best) for x in preds_test]).tolist()
    t2 = time.time()
    print(f"Usedtime for prediction = {t2-t1} seconds")

    image_grain = un.merge_images(preds_test)*128
    image = cv2.imread(os.path.join(output_path, fn_img), cv2.IMREAD_GRAYSCALE)
    # img_size = image.shape
    image_temp1 = cv2.inRange(image, 0, 0)
    image_temp1[np.nonzero(image_temp1)] = 0   # other

    image_temp2 = cv2.inRange(image, threshold[3][1], threshold[3][1])  #threshold[3][1]
    image_temp2[np.nonzero(image_temp2)] = 255   # clay and matrix
    image_mineral = image_temp1 + image_temp2

    image1 = image_mineral + image_grain
    image_grain = cv2.inRange(image1, 383, 383)   # np.uint8([383]), np.uint8([383])
    image_grain[np.nonzero(image_grain)] = threshold[3][1]    # grain

    image_clay = cv2.inRange(image1, 255, 255)
    image_clay[np.nonzero(image_clay)] = threshold[3][0]    # clay

    cv2.imwrite(os.path.join(output_path, 'img_Clay.tif'), image_clay)
    cv2.imwrite(os.path.join(output_path, 'img_Grain.tif'), image_grain)
    gc.collect()
    # plt.show()


#-----------------------------------------------------------------------------------------------------------------------
def main():
    start_time = time.time()
    # current_path = os.path.dirname(__file__)
    input_path = os.path.join(current_path, prj_fold)
    done_path = os.path.join(input_path, 'done')
    os.makedirs(done_path, exist_ok=True)
    idx = 0
    for file in sorted(os.listdir(input_path)):
        if file.endswith(".tif"):
            input_file = os.path.join(input_path,file)
            output,_ = os.path.splitext(file)
            output_path = os.path.join(input_path, output)
            os.makedirs(output_path, exist_ok=True)

            ip.image_process_1(input_file, output_path, threshold)
            prediction_unet(idx,input_file,input_path,output_path, 'img_Matrix.tif', threshold)
            ip.calculate_component(output_path, 'img_Pore.tif', 'img_Organic.tif', 'img_Grain.tif', 'img_Clay.tif',
                                   'img_Pyrite.tif', threshold)
            idx += 1
            shutil.move(input_file, os.path.join(done_path, file))
            print(file,' is Done\n')

    time_dif = time.time() - start_time
    print('\nTime Used: ' + str(timedelta(seconds=int(round(time_dif)))), '\n')


if __name__ == '__main__':
    main()
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main=main())

