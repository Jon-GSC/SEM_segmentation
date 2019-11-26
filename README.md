# SEM images segmentation with Tensorflow

Scanning Electron Microscope (SEM) image segmentation.
The aim of this SEM image segmentation study is to identify and analyse geological source rock objects using deep learning.    


### Requirements: 

* Install python 3.6/newer and anaconda packages: conda install keras, tensorflow, scikit-image, opencv, tqdm, pandas, numpy, seaborn libraries.


### Dataset: 

* The original image is out allowed size limit. The loaded images are used for demonstration purpose, the default image size is 256x256. During the trainint, the images in train and mask folder are split into training and validation groups. The file structure is:
'''
       img_path
       |---train_image
       |   |---*.tif
       |---mask_image
       |   |---*.tif
       |---raw_image
       |   |---output
       |       |---*.tif
'''


### Instruction:

  1. The two main python codes and /itools need be save in one folder.

  2. forWeb_SEM_Model_1.py are used to train the model, the model is saved automatically after validation. Some parameters at the beginning of code are default, user can change it as wish.

  3. forWeb_SEM_clay_analysis.py are used to load the created model, and used it to predict the features of new images. All of the images should be saved in one folder, code will run through each of the image and save all of the results in new folder individually.


### Hardware used: 

* HP-7920 workstation: 56core CPU; 64G memory; GPU Nvidia Quadro P5000.


### Contact: 

* jon.liu@canada.ca


### Acknowledgments

* SEM image segmentation used open source codes and library from github, google, kaggle, csdn blog.


