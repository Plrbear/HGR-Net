import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import pickle
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import *
from keras.objectives import *
from keras.metrics import binary_accuracy
from keras.models import load_model
import keras.backend as K
from model import SegModel
from dataLoader import dataloader
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score
#import keras.utils.visualize_util as vis_util
import argparse
from keras import optimizers

from utils.SegDataGenerator import *
import time


# class SegTrainer(object):
#         def __init__(self,opt):
#             self.opt=opt
            
def train(opt):  
    
    #####Compile mode####
    input_size=(opt.row,opt.col,opt.ch)
    SegM=SegModel(input_size)
    model=SegM.model
    _adam=optimizers.Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='binary_crossentropy',optimizer = _adam,metrics=['accuracy'])
    ###################load data###########
    img_dim=(opt.row,opt.col)
    [train_img,train_mask,test_img,test_mask]=dataloader(opt.ad1, opt.ad2, opt.ad3, opt.ad5, opt.img_format, img_dim)
    ####check point####
    model_checkpoint = ModelCheckpoint(opt.chekp+'.hdf5', monitor='val_acc',verbose=1, save_best_only=True)

    ###################train###########
    hist=model.fit(train_img, train_mask,validation_data=(test_img,test_mask), batch_size=opt.batch_size, nb_epoch=opt.epochs, verbose=1,callbacks=[model_checkpoint])
 #    ######evaluate#####
    model.load_weights(opt.chekp+'.hdf5')
    y_pred=model.predict(test_img)
    f=fscore(y_pred,test_img,test_mask)
    print(f)
    return hist,f 

def fscore(tp,Images,Masks):
        

    total=0
    i=0
    fs=0
    for i in range(len(Images)):
        total += 1


        tp[i][tp[i]>0.5]=1
        tp[i][tp[i]<0.5]=0

        pred = img_to_array(tp[i]).astype(int)
        label = img_to_array(np.squeeze(Masks[i], axis=2)).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)

        fs += f1_score(flat_label,flat_pred, average='micro')

    fs=fs/total

    return fs  
            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--input_size', type=list, default='(320,320,3)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--ad1', type=str, default='')
    parser.add_argument('--ad2', type=str, default='')
    parser.add_argument('--ad3', type=str, default='')
    parser.add_argument('--ad5', type=str, default='')
    parser.add_argument('--img_format', type=str, default='*.png')
    parser.add_argument('--chekp', type=str, default='')
    parser.add_argument('--row', type=int, default=320)
    parser.add_argument('--col', type=int, default=320)
    parser.add_argument('--ch', type=int, default=3)


    opt = parser.parse_args()
    train(opt)            
