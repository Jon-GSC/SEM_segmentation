''' Tools for prepare images for machine learning '''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def slice_image(img, x_size, y_size):
    images = []
    y = 0
    while y < img.shape[0]:
        y_end = min(y + y_size, img.shape[0])
        x = 0
        while x < img.shape[1]:
            x_end = min(x + x_size, img.shape[1])
            images.append(img[x:x_end, y:y_end])
            x = x_end
        y = y_end
    return images

def batch_write(file_name, images):
    filename, ext = os.path.splitext(file_name)
    for image, count in zip(images, range(len(images))):
        name = "{}.{:010}{}".format(filename, count, ext)
        cv2.imwrite(name, image)

def slice_images(in_path, out_path, width, height):
    os.makedirs(out_path, exist_ok=True)
    for file in sorted(os.listdir(in_path)):
        if file.endswith(".tif"):
            image = cv2.imread(os.path.join(in_path, file), cv2.IMREAD_GRAYSCALE)
            images = slice_image(image, width, height)
            batch_write(os.path.join(out_path, file), images)

def remove_empty(empty_file_path, relative_file_path, minimum_ratio = 0.1):
    for file in os.listdir(empty_file_path):
        if file.endswith(".tif"):
            empty_file = os.path.join(empty_file_path, file)
            other_file = os.path.join(relative_file_path, file)
            image = cv2.imread(empty_file, cv2.IMREAD_GRAYSCALE)
            hist = np.histogram(image.flatten(), 256, [0, 256])[0]
            if hist[255]/(hist[0]+hist[255]) < minimum_ratio:
                os.remove(empty_file)
                os.remove(other_file)

def merge_images(images,filename):  #merge
    images = np.array(images)
    n_img = len(images)
    size_img = images[0].shape
    n_img_0 = int(np.sqrt(n_img))
    if n_img_0 != np.floor(n_img_0):
        print('Error!, amount of images is not square!')
    # np.reshape(images,(None,10*128))
    image = images.reshape(n_img_0,n_img_0,size_img[0],size_img[1]).transpose(1,0,2,3).swapaxes(1,2).reshape(size_img[0]*n_img_0,-1)
    # print(image[0:30,0:30],'\n',type(image),image.shape, n_img, size_img)

    fig88 = plt.figure(88)
    plt.imshow(image, cmap='gray')
    fig88.suptitle('Merged image')
    # plt.savefig(filename)
    # plt.show()
    image[np.nonzero(image)] = 255
    image = np.uint8(image)  # to gray
    cv2.imwrite(filename, image)

    clay_pixel = np.count_nonzero(image==255)
    total_pixel = np.prod(image.shape)
    print('Predicted image saved. \nThe percentage of Clay/organic is: ',100*clay_pixel/total_pixel,'%\n')
