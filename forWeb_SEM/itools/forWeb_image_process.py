# general processing sem image by Jon on 2018-06-28
# cnn works after adjust learning_rate etc parameters, estimator.train used and work good.
#
# NOTE: some functions are not called, for testing only, ignore those.
#
import os
import cv2
import collections
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import signal
# from multiprocessing import process
# from win64api import GetSystemMetrics
#-----------------------------------------------------------------------------------------------------------------------
learning_rate = 0.0001   # estimator works,better than cnn. on 20180711

batch_size = 80
display_step = 100

n_hidden_1 = 256
n_hidden_2 = 128
num_classes = 3
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
def image_process_1(input_file,output_path,threshold):

    kernel = np.ones((3, 3), np.float32) / 10
    kernel[1, 1] = 0.2
    # for file in os.listdir(input_path):
    #     if file.endswith(".tif"):
    image = cv2.imread(os.path.join(input_file), cv2.IMREAD_GRAYSCALE)
    # image1 = cv2.filter2D(image, -1, kernel)  # smooth.
    image1 = cv2.medianBlur(image, 5)  # median filtering
    # image1 = cv2.bilateralFilter(image,5,115,115)  # bilateral filtering

    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0

    image_por = cv2.inRange(image1, threshold[1][0]-1, threshold[1][1])
    image_por[np.nonzero(image_por)] = threshold[1][0]
    # image_por = cv2.morphologyEx(image_por,cv2.MORPH_CLOSE,kernel)

    image_org = cv2.inRange(image1, threshold[2][0], threshold[2][1])
    image_org[np.nonzero(image_org)] = threshold[2][0]
    # image_org = cv2.morphologyEx(image_org,cv2.MORPH_CLOSE,kernel)

    image_mat = cv2.inRange(image1, threshold[3][0], threshold[3][1])
    image_mat[np.nonzero(image_mat)] = 1 #threshold[3][1]
#    image_mat = cv2.morphologyEx(image_mat, cv2.MORPH_CLOSE, kernel)   # remove small pixel...
    image_mat1 = signal.convolve2d(image_mat, kernel, mode='same', fillvalue=1)  # remove single pixel
    image_mat2 = image_mat.copy()
    image_mat2[np.logical_and(image_mat1 == 8, image_mat == 0)] = threshold[3][1]
    image_mat = image_mat2.copy()
    image_mat[np.nonzero(image_mat)] = threshold[3][1]

    image_pyr = cv2.inRange(image1, threshold[4][0], threshold[4][1])
    image_pyr[np.nonzero(image_pyr)] = threshold[4][1]
    # image_pyr = cv2.morphologyEx(image_pyr,cv2.MORPH_CLOSE,kernel)

    image_por[image_por+image_mat==np.uint8(threshold[1][0]+threshold[3][1])] = 0
    image_org[image_org+image_mat==np.uint8(threshold[2][0]+threshold[3][1])] = 0
    image_pyr[image_pyr+image_mat==np.uint8(threshold[4][1]+threshold[3][1])] = 0

    image2 = image_por + image_org + image_mat + image_pyr

    fig1 = plt.figure(1)  # orginal image
    plt.imshow(image1, cmap='gray')
    fig1.suptitle('Input image')

    fig2 = plt.figure(2)  # Pyrite image
    plt.imshow(image_pyr, cmap='gray')
    fig2.suptitle('Pyrite')

    fig3 = plt.figure(3)  # Matrix image
    plt.imshow(image_mat, cmap='gray')
    fig3.suptitle('Matrix')

    fig4 = plt.figure(4)  # Organic image
    plt.imshow(image_org, cmap='gray')
    fig4.suptitle('Organic')

    fig5 = plt.figure(5)  # Porosity image
    plt.imshow(image_por, cmap='gray')
    fig5.suptitle('Pore')

    fig6 = plt.figure(6)
    plt.imshow(image2, cmap=plt.cm.get_cmap('jet', 4))  # jet
    plt.colorbar()
    fig6.suptitle('Classified')
    # plt.show()

    cv2.imwrite(os.path.join(output_path, 'img_RawImg.tif'), image1)
    cv2.imwrite(os.path.join(output_path, 'img_Pyrite.tif'), image_pyr)
    cv2.imwrite(os.path.join(output_path, 'img_Matrix.tif'), image_mat)
    cv2.imwrite(os.path.join(output_path, 'img_Organic.tif'), image_org)
    cv2.imwrite(os.path.join(output_path, 'img_Pore.tif'), image_por)

    cv2.imwrite(os.path.join(output_path, 'img_Classfied_1.tif'), image2)
    # plt.imsave(os.path.join(output_path, 'img_Classfied_1.tif'), image2, cmap=plt.cm.get_cmap('jet', 4))

    pyrite_pixel = np.count_nonzero(image_pyr != 0)
    matrix_pixel = np.count_nonzero(image_mat != 0)
    organic_pixel = np.count_nonzero(image_org != 0)
    pore_pixel = np.count_nonzero(image_por != 0)
    total_pixel = np.prod(image.shape)

    print('\nworking on ', input_file)
    print('\nPyrite: {:2.2f}%'.format(100 * pyrite_pixel / total_pixel))
    print('Matrix: {:2.2f}%'.format(100 * matrix_pixel / total_pixel))
    print('Organic:{:2.2f}%'.format(100 * organic_pixel / total_pixel))
    print('Pore:   {:2.2f}%\n'.format(100 * pore_pixel / total_pixel))


def gen_training_dataset(input_path0,input_path1,input_pre_path,output_path,threshold,n_hw):  # organic pore or matrix pore / MNIST way
    #     opore  /   organic  /   matrix  /   pyrite
    n_height = n_hw[0]
    n_width = n_hw[1]
    temp = os.path.join(output_path, 'temp')
    pred_path = os.path.join(output_path, 'predict')

    inputs = __image_load(output_path, img_ext='tif')
    targets = __image_load(temp, img_ext='tif')
    predicts = __image_load(pred_path, img_ext='tif')
    x = np.array(inputs)
    x0 = np.array(predicts)
    y = np.zeros(shape=(len(inputs),3),dtype=np.int)
    z = np.zeros(shape=len(inputs),dtype=np.int)
    for i in range(len(inputs)):
        p_pyrite = np.count_nonzero(targets[i] == threshold[4][1])
        p_matrix = np.count_nonzero(targets[i] == threshold[3][1]) #
        p_organic = np.count_nonzero(targets[i] == threshold[2][1])
        p_pore = np.count_nonzero(targets[i] == threshold[1][1])
        if p_pyrite>=int(0.5*n_height*n_width):   # image_mat[np.nonzero(image_mat)] = 255  np.count_nonzero(image_pyr != 0)
            y[i] = [0, 0, 1]
            z[i] = 2
        if p_pyrite<int(0.5*n_height*n_width):
            if p_matrix >= int(0.85*n_height*n_width):
                y[i] = [0, 1, 0]
                z[i] = 1
            if p_matrix < int(0.85*n_height*n_width):   # need update,
                y[i] = [1, 0, 0]
                z[i] = 0

    threshold0 = int(0.9 * len(x))
    train = (np.stack(x[:threshold0]), np.stack(y[:threshold0]), np.stack(z[:threshold0]))
    evaluate = (np.stack(x[threshold0:]), np.stack(y[threshold0:]), np.stack(z[threshold0:]))
    predict = (np.stack(x0[0:]),0)   #tuple must have at least two, so add 0
    # save like MNIST
    # print('check: ',type(predict),type(evaluate),evaluate[0][0:5],predict[0][0:5])
    return train,evaluate,predict


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
            if img.shape[0] == img.shape[1]:
                img = np.array(img).reshape(-1)#.tolist()
                images.append(img)
    return images


#-----------------------------------------------------------------------------------------------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def deeplearning(output_path,train, evaluate, predict, iteration):
# load tensorflow

    size_img = predict[0][0].shape
    sess = tf.InteractiveSession()
    pred_path = os.path.join(output_path, 'predict')
    x = tf.placeholder(tf.float32,[None,size_img[0]])
    y = tf.placeholder(tf.float32,[None,3])
    keep_prob = tf.placeholder(tf.float32)

    x_image = tf.reshape(x,[-1,int(np.sqrt(size_img)),int(np.sqrt(size_img)),1])

    W_conv1 = weight_variable([7,7,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([7,7,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([16*16*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool3 = tf.reshape(h_pool2,[-1,16*16*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3,W_fc1)+b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    W_fc2 = weight_variable([1024,3])
    b_fc2 = bias_variable([3])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_conv),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy) #GradientDescent
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    predict0 = tf.argmax(y_conv,1)
    predict1 = tf.argmax(y,1)
    tf.global_variables_initializer().run()

    batch0 = (1/(1+np.exp(-evaluate[0]/255)), evaluate[1], evaluate[2])     #
    batch1 = (1/(1+np.exp(-predict[0]/255)), 0)     #
    for i in range(iteration):
        idx = np.random.randint(0, train[1].shape[0], batch_size)
        # idx = np.array([7389,7390,7391,7392,7393,7394,7395,7398,7399,7400])
        batch = (1 / (1 + np.exp(-train[0][idx] / 255)), train[1][idx], train[2][idx])  #
        # for i in range(iteration):
            # print(batch[0][0:],'\n',len(batch),batch[0].shape,batch[0][0].shape, batch[1].shape,'\n',batch)
            # plt.imshow(np.reshape(batch[0][0], [64, -1]), cmap='gray')
            # plt.show()
            # exit()
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1})
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        # prediction0 = y_conv.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        if i % 200 == 0:
            print('step %d,training accuracy %g ' %(i,train_accuracy))
    temp0 = predict0.eval(feed_dict={x:batch0[0],keep_prob:1.0})   #,y:batch0[1]
    temp1 = predict1.eval(feed_dict={y:batch0[1],keep_prob:1.0})   #x:batch0[0],
    correct_prediction = np.sum(temp0 == temp1)  # need check output.
    accuracy = 100 * correct_prediction / temp0.shape[0]
    print('Evaluation results:')
    print('\nreal:\n',temp1[:200],'\npreds:\n', temp0[:200], '\ni:{}, accuracy: {}%'.format(i,accuracy))

    preds = np.zeros(batch1[0].shape[0])
    for i in range(16):  #range(batch1[0].shape[0]):
        preds[i*576:(i+1)*576] = predict0.eval(feed_dict={x:batch1[0][i*576:(i+1)*576],keep_prob:1.0})  #
    return preds
    # print('test accuracy %g' %accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))


#-----------------------------------------------------------------------------------------------------------------------
def neural_net(x_dict):
    x = x_dict['images']
    layer_1 = tf.layers.dense(x,n_hidden_1)
    layer_2 = tf.layers.dense(layer_1,n_hidden_2)
    out_layer = tf.layers.dense(layer_2,num_classes)
    return out_layer


def model_fn(features,labels,mode):
    logits = neural_net(features)
    pred_classes = tf.argmax(logits,axis=1)
    pred_probas = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=pred_classes)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.cast(labels,dtype=tf.int32)))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(mode=mode,
                                             predictions=pred_classes,
                                             loss=loss_op,
                                             train_op=train_op,
                                             eval_metric_ops={'accuracy':acc_op})
    print('feature:', features, '\nlabel:',labels,'\nmode:', mode)
    return estim_specs


def deeplearning_0(train, evaluate, predict, num_steps):
# load tensorflow
    num_input = predict[0].shape
    n_images = 200   # for evaluating
    # idx = np.random.randint(0, train[1].shape[0], batch_size)

    batch = (train[0]/255, train[1], train[2])  # 1/(1+np.exp(-train[0][idx]/255))
    batch0 = (evaluate[0]/255, evaluate[1], evaluate[2])  # 1/(1+np.exp(-evaluate[0][idx]/255))
    batch1 = (predict[0]/255,0)  #tulip
    # print(batch0[0].shape,batch1[0].shape,'\n',batch0[0][0:2],'\n',num_input)   # batch1[0][0:2]

    for i in range(5):
        input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':batch[0]},y=batch[2],batch_size=batch_size,num_epochs=None,shuffle=True)
        model = tf.estimator.Estimator(model_fn)
        model.train(input_fn,steps=num_steps)    # training

        input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': batch0[0][:n_images]}, shuffle=False)
        pred_eval = np.array(list(model.predict(input_fn)))   # evaluating
        correct_prediction = np.sum(pred_eval == batch0[2][:n_images])  # need check output.
        accuracy = 100 * correct_prediction / pred_eval.shape[0]
        print('Evaluation results:')
        print('i:',i,'\nreal:\n', batch0[2][:n_images], '\npreds:\n', pred_eval, '\naccuracy: {}%'.format(accuracy))

        input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':batch1[0]},shuffle=False)
        preds = np.array(list(model.predict(input_fn)))    # predicting
    print(preds)
    return preds


def image_analysis(output_path,predict,preds,threshold):

    print(predict[0].shape,type(predict))

    pred_path = os.path.join(output_path, 'predict')
    os.makedirs(pred_path, exist_ok=True)
    size_img = predict[0].shape[0]
    hw_img = predict[0].shape[1]
    img_group = np.zeros(predict[0].shape)
    for i in range(size_img):
        img_temp = predict[0][i]
        image_por = cv2.inRange(img_temp, threshold[1][0]-1, threshold[1][1])
        if preds[i]==0 or preds[i]==2:
            image_por[np.nonzero(image_por)] = 0
        else:
            image_por[np.nonzero(image_por)] = 60

        image_org = cv2.inRange(img_temp, threshold[2][0], threshold[2][1])
        image_org[np.nonzero(image_org)] = 110

        image_mat = cv2.inRange(img_temp, threshold[3][0], threshold[3][1])
        if preds[i] == 0:
            image_mat[np.nonzero(image_mat)] = 160
        elif preds[i] == 1:
            image_mat[np.nonzero(image_mat)] = 200

        image_pyr = cv2.inRange(img_temp, threshold[4][0], threshold[4][1])
        image_pyr[np.nonzero(image_pyr)] = 255

        image0 = image_por + image_org + image_mat + image_pyr
        img_group[i,:] = np.transpose(image0)

    ni = int(np.sqrt(size_img))
    nhw = int(np.sqrt(hw_img))
    img_group = np.reshape(img_group, [ni, ni, nhw,-1]).transpose(1,0,2,3).swapaxes(1,2).reshape(ni*nhw,-1)
    cv2.imwrite(os.path.join(pred_path, 'img_Outcome.tif'), img_group)
    # print(type(img_group),img_group.shape)
    fig1 = plt.figure(1)
    plt.imshow(img_group, cmap=plt.cm.get_cmap('jet', 256))  # jet
    plt.colorbar()
    fig1.suptitle('AI_Classified')
    plt.show()
    plt.imsave(os.path.join(pred_path, 'img_Outcome_color.tif'), img_group, cmap=plt.cm.get_cmap('jet', 256))


# @numba.jit(nopython=True,parallel=True)
def clay_index(x0,y0,img):
    n_area = 64
    # n_size = img.shape
    num_surround = np.zeros([n_area])

    idx0 = np.argwhere(img == 0)
    angle0 = np.arctan2((idx0[:, 1] - y0), (idx0[:, 0] - x0))
    angle0[angle0 < 0] += 2 * np.pi
    idx_group = np.floor(angle0 / (2 * np.pi / n_area))
    num_count = collections.Counter(idx_group)
    num_key = np.array(list(num_count.keys())).astype(int)
    num_value = np.array(list(num_count.values())).astype(int)
    num_surround[num_key] = num_value

    num_surround1 = num_surround.copy()
    num_surround1[num_surround1 != 0] = 1

    n_nonzeros = np.count_nonzero(num_surround != 0)
    n_per = n_nonzeros / n_area

    clay_index = 0
    s1 = np.dot(num_surround1, np.concatenate((num_surround1[int(n_area/2+1):], num_surround1[0:int(n_area/2+1)]), axis=0))
    s2 = np.dot(num_surround1, num_surround1)

    bool_clay = n_per>0.7 and s1/s2>0.099 and max(num_surround)>=3
    if bool_clay:   #clay
        clay_index = 128
        # print(s1, ' _ ', s2, ' _ ', s1 / s2)
    return clay_index  #n_per, c_per


def classify_clay(output_path,fn_img,threshold):
    # note: using circle analysis/separate the clay and matrix mineral . Jon on 2018-07-16
    #
    n_pix = 1
    xy_radium = 41
    image = cv2.imread(os.path.join(output_path, fn_img), cv2.IMREAD_GRAYSCALE)
    img_size = image.shape
    image_temp1 = cv2.inRange(image, 0, 0)
    image_temp1[np.nonzero(image_temp1)] = 0   # other
    image_temp2 = cv2.inRange(image, threshold[3][1], threshold[3][1])
    image_temp2[np.nonzero(image_temp2)] = 255   # clay and matrix
    image_mineral = image_temp1 + image_temp2

    img_clay = np.zeros(img_size,np.uint8)
    # grain_clay = 999*np.ones([img_size, 2])
    istep = n_pix//2

    y_centre = istep
    while y_centre < img_size[0]:
        y0 = min([y_centre, xy_radium])
        y1 = max(y1 for y1 in [0, y_centre - xy_radium] if y1 >= 0)
        y2 = min(y2 for y2 in [y_centre + xy_radium + 1, img_size[0] + 1] if y2 <= img_size[0] + 1)
        x_centre = istep
        while x_centre < img_size[1]:
            if image_mineral[x_centre,y_centre]==255:
                # img = np.zeros([2*xy_radium+1,2*xy_radium+1])   # get surround img
                x0 = min([x_centre,xy_radium])
                x1 = max(x1 for x1 in [0, x_centre-xy_radium] if x1 >= 0)
                x2 = min(x2 for x2 in [x_centre+xy_radium+1, img_size[1]+1] if x2 <= img_size[1]+1)
                img = image_mineral[x1:x2,y1:y2]
                # print(img[xy_radium,xy_radium],image_mineral[x_centre,y_centre])
                clay_index0 = clay_index(x0,y0,img)   #get clay or mineral, #,n_per, c_per,
                img_clay[x_centre - istep:x_centre + istep + 1, y_centre - istep:y_centre + istep + 1] = clay_index0

            x_centre += n_pix
        y_centre += n_pix
        if y_centre%10==0:
            print('finished: {0:.2f}%'.format(100*y_centre/img_size[0]))

    kernel = np.ones((5,5),np.uint8)
    img_clay = cv2.dilate(img_clay,kernel,iterations=2)
    img_clay = cv2.medianBlur(img_clay, 5)
    img_clay = cv2.erode(img_clay,kernel,iterations=2)

    # img_mat = cv2.inRange(image, threshold[3][1], threshold[3][1])
    image1 = image_mineral + img_clay  # 255 + 128
    image_clay = cv2.inRange(image1, np.uint8([383]), np.uint8([383]))
    image_clay[np.nonzero(image_clay)] = threshold[3][0]    #clay

    # image2 = image_mineral + image_clay
    image_grain = cv2.inRange(image1, 255, 255)
    image_grain[np.nonzero(image_grain)] = threshold[3][1]    #grain
    # image_targe = image_targe1 + image_targe2

    cv2.imwrite(os.path.join(output_path, 'img_Clay.tif'), image_clay)
    cv2.imwrite(os.path.join(output_path, 'img_Grain.tif'), image_grain)


# @jit(nopython=True,parallel=True)    #cuda.jit  #cuda.jit('void(int32[:],int32[:],int32[:],float32[:])',device=False)  # nogil=True
def test_loop(n_pix,xy_radium,img_size,image_mineral):
    n_area = 64
    img_clay = np.zeros(img_size,np.uint8)    #img_clay = np.empty_like(image_mineral)  #
    istep = round(n_pix/2)
    # grain_clay = np.zeros([img_size,3])

    # y_centre = istep
    n_xpoint = round(img_size[1] / n_pix)
    n_ypoint = round(img_size[0] / n_pix)
    for iy in range(n_ypoint):  # while y_centre < img_size[0]:  #numba.prange(istep,img_size[0],n_pix):
        y_centre = iy * n_pix
        # start = datetime.datetime.now(dtype=np.int64)
        y0 = np.min(np.array([y_centre, xy_radium]))
        y1 = np.max(np.array([0, y_centre - xy_radium])) #np.max(y1 for y1 in np.array([0, y_centre - xy_radium]) if y1 >= 0)
        y2 = np.min(np.array([y_centre + xy_radium + 1, img_size[0] + 1]))  #np.min(y2 for y2 in np.array([y_centre + xy_radium + 1, img_size[0] + 1]) if y2 <= img_size[0] + 1)
        # x_centre = istep
        for ix in range(n_xpoint):
            x_centre = ix * n_pix
            if image_mineral[x_centre,y_centre]==255:
                # img = np.zeros([2*xy_radium+1,2*xy_radium+1])  # get surround img
                x0 = np.min(np.array([x_centre,xy_radium]))
                x1 = np.max(np.array([0, x_centre-xy_radium]))  #np.max(x1 for x1 in [0, x_centre-xy_radium] if x1 >= 0)
                x2 = np.min(np.array([x_centre+xy_radium+1, img_size[1]+1]))  #np.min(x2 for x2 in [x_centre+xy_radium+1, img_size[1]+1] if x2 <= img_size[1]+1)
                # print(type(x1),type(x2),type(y1),type(y2))
                img = image_mineral[x1:x2,y1:y2]

                idx0 = np.argwhere(img == 0)
                num_surround = np.zeros(n_area)
                angle0 = np.arctan2((idx0[:, 1] - y0), (idx0[:, 0] - x0))
                angle0[angle0 < 0] += 2 * np.pi
                idx_group = np.floor(angle0 / (2 * np.pi / n_area))
                num_count = collections.Counter(idx_group)
                num_key = np.array(list(num_count.keys())).astype(int)
                num_value = np.array(list(num_count.values())).astype(int)
                num_surround[num_key] = num_value
                # print(num_surround)
                n_nonzeros = np.count_nonzero(num_surround != 0)
                #n_nonzeros = len(np.nonzero(num_surround)[0])
                s1 = np.dot(num_surround, np.concatenate((num_surround[int(n_area / 2 + 1):], num_surround[0:int(n_area / 2 + 1)]), axis=0))
                s2 = np.dot(num_surround, num_surround)
                # grain_clay[i,j,0] =  n_nonzeros / n_area

                clay_index0 = 0
                bool_clay = n_nonzeros / n_area > 0.59 and s1 / s2 > 0.099 and np.max(num_surround) >= 5
                if bool_clay:  # clay
                    clay_index0 = 255
                    # grain_clay[i, j, 2] = 1
                    # print(s1, ' _ ', s2, ' _ ', s1 / s2)
                img_clay[x_centre - istep:x_centre + istep + 1, y_centre - istep:y_centre + istep + 1] = clay_index0
        if y_centre%10==0:
            print((100*y_centre/img_size[0]),) #  'finished: {0:.2f}%'.format
    return img_clay


def classify_clay_GPU(output_path,fn_img,threshold0):  #
    #note: using circle analysis/separate the clay and matrix mineral . Jon on 2018-07-16
    # threshold = {1:[1, 110],2:[111, 160],3:[161, 199],4:[200, 255]}
    n_pix = 7
    xy_radium = 39
    image = cv2.imread(os.path.join(output_path, fn_img), cv2.IMREAD_GRAYSCALE)
    img_size = image.shape
    image_temp1 = cv2.inRange(image, 0, 0)
    image_temp1[np.nonzero(image_temp1)] = 0   # other
    image_temp2 = cv2.inRange(image, 199,199)  #threshold0[2,1], threshold0[2,1]
    image_temp2[np.nonzero(image_temp2)] = 255   # clay and matrix
    image_mineral = image_temp1 + image_temp2

    img_clay = test_loop(n_pix, xy_radium, img_size, image_mineral)
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     img_clay = sess.run(img_clay)

    # img_mat = cv2.inRange(image, threshold[3][1], threshold[3][1])
    # img_mat[np.nonzero(img_mat)] = 255
    image1 = image_mineral + img_clay
    image_clay = cv2.inRange(image1, 510, 510)
    image_clay[np.nonzero(image_clay)] = threshold0[2,0]    #clay

    # image2 = image_mineral + image_clay
    image_grain = cv2.inRange(image1, 255, 255)
    image_grain[np.nonzero(image_grain)] = threshold0[2,1]    #grain
    # image_targe = image_targe1 + image_targe2

    cv2.imwrite(os.path.join(output_path, 'img_Clay.tif'), image_clay)
    cv2.imwrite(os.path.join(output_path, 'img_Grain.tif'), image_grain)


def calculate_component(output_path,fn_img_pore,fn_img_organic,fn_img_grain,fn_img_clay,fn_img_pyrite,threshold):
    image_por = cv2.imread(os.path.join(output_path, fn_img_pore), cv2.IMREAD_GRAYSCALE)
    image_org = cv2.imread(os.path.join(output_path, fn_img_organic), cv2.IMREAD_GRAYSCALE)
    image_grain = cv2.imread(os.path.join(output_path, fn_img_grain), cv2.IMREAD_GRAYSCALE)
    image_clay = cv2.imread(os.path.join(output_path, fn_img_clay), cv2.IMREAD_GRAYSCALE)
    image_pyr = cv2.imread(os.path.join(output_path, fn_img_pyrite), cv2.IMREAD_GRAYSCALE)

    image_all = image_por + image_org + image_grain + image_clay + image_pyr

    cv2.imwrite(os.path.join(output_path, 'classfied_All.tif'), image_all)
    plt.imsave(os.path.join(output_path, 'classfied_All(color).tif'), image_all, cmap=plt.cm.get_cmap('jet', 256))
    #
    total_pixel = np.prod(image_all.shape)
    pyrite_pixel = np.count_nonzero(image_pyr != 0) * 100 / total_pixel
    matrix_pixel = np.count_nonzero(image_grain != 0) * 100 / total_pixel
    clay_pixel = np.count_nonzero(image_clay != 0) * 100 / total_pixel
    organic_pixel = np.count_nonzero(image_org != 0) * 100 / total_pixel
    pore_pixel = np.count_nonzero(image_por != 0) * 100 / total_pixel
    print('\nPyrite: {:2.2f}%'.format(pyrite_pixel))
    print('Grain: {:2.2f}%'.format(matrix_pixel))
    print('Clay: {:2.2f}%'.format(clay_pixel))
    print('Organic:{:2.2f}%'.format(organic_pixel))
    print('T.Pore:   {:2.2f}%'.format(pore_pixel))

    fid = open(os.path.join(output_path,'Results.txt'), 'w')
    fid.write('Pixel Percentage:\n')
    fid.write('Pyrite: {:2.2f}%\n'.format(pyrite_pixel))
    fid.write('Grain: {:2.2f}%\n'.format(matrix_pixel))
    fid.write('Clay: {:2.2f}%\n'.format(clay_pixel))
    fid.write('Organic:{:2.2f}%\n'.format(organic_pixel))
    fid.write('T.Pore: {:2.2f}%\n'.format(pore_pixel))
    fid.close()

    dirName = os.path.basename(output_path)
    fid1 = open(os.path.join(os.path.abspath(os.path.join(output_path, '..')),'Result_all.txt'),'a')
    fid1.write('Sample_Name        Pyrite   Grain   Clay   Organic   T.Pore\n{:s}  {:2.2f}%  {:2.2f}%  {:2.2f}%  {:2.2f}%  {:2.2f}%\n'
               .format(dirName,pyrite_pixel,matrix_pixel,clay_pixel,organic_pixel,pore_pixel))
    fid1.close()

