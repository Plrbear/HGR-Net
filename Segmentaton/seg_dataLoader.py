import os
from glob import glob
import matplotlib.pyplot as plt
from os import listdir
from PIL import Image as PImage
import numpy as np
import pandas as pd
from scipy import ndimage, misc
from PIL import Image

from skimage.io import imread

def dataloader(ad1,ad2,ad3,ad5,img_format,img_dim):
#ad1: address of a fold ( among 5 folds for cross validation)
#ad2: name of train images folder 
#ad3: name of test images folder 
#ad5: name of the mask folder
#ie.

# ad1='/home/amir/epi/s2/'
# ad2='st'
# ad4='s2/st'
# ad5='smask'
# format_='*.png'
# img_dim=(320,320)




    fold_name=os.path.basename(os.path.normpath(ad1))
    ad4=fold_name+'/'+ad2
    ad6=fold_name+'/'+ad3
    #####################################train#########################
    BASE_IMAGE_PATH = os.path.join('..', ad1)
    all_images = sorted(glob(os.path.join(BASE_IMAGE_PATH, ad2, img_format)))
    ################test
    BASE_IMAGE_PATH = os.path.join('..', ad1)
    all_imagest = sorted(glob(os.path.join(BASE_IMAGE_PATH, ad3, img_format)))
    ##########data loader train
    import re
    TrainImages = []
    TrainMasks = []
    for i in all_images:
        image = ndimage.imread(i, mode='RGB')
        j = re.sub(ad4, ad5, i)
#         k = re.sub('.tif', '.png', j)
        mask = ndimage.imread(j , mode='L' )
        image_resized = misc.imresize(image, img_dim)
        mask_resized = misc.imresize(mask, img_dim)
        TrainImages.append(image_resized)
        TrainMasks.append(mask_resized)   
    #################### data loader test

    TestImages = []
    TestMasks = []
    for i in all_imagest:
        image = ndimage.imread(i, mode='RGB')
        j = re.sub(ad6, ad5, i)
#         k = re.sub('tif', 'png', j)
        mask = ndimage.imread(j , mode='L' )
        image_resized = misc.imresize(image, img_dim)
        mask_resized = misc.imresize(mask, img_dim)
        TestImages.append(image_resized)
        TestMasks.append(mask_resized)   
    #############pre
    TrainImages=np.array(TrainImages)
    TrainImages=TrainImages.astype('float32')
    TrainImages /= 255.0
    TestImages=np.array(TestImages)
    TestImages=TestImages.astype('float32')
    TestImages /= 255.0
    ########
    TrainMasks=np.array(TrainMasks)
    TestMasks=np.array(TestMasks)
    TrainMasks[TrainMasks>0]=255
    TrainMasks[TrainMasks<=0]=0
    TestMasks[TestMasks>0]=255
    TestMasks[TestMasks<=0]=0
    TrainMasks=TrainMasks.astype('float32')
    TrainMasks /= 255.0
    TestMasks=TestMasks.astype('float32')
    TestMasks /= 255.0
    TrainMasks=np.expand_dims(TrainMasks,axis=3)
    TestMasks=np.expand_dims(TestMasks,axis=3)
    #############
    return TrainImages,TrainMasks,TestImages,TestMasks




