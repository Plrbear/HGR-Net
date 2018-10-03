from sklearn.feature_extraction import image
import os
import numpy as np
from PIL import Image
import random

from keras.layers import Dense, Dropout, Activation, \
                         Flatten, Convolution2D, MaxPooling2D, \
                         BatchNormalization, Conv2D, Input,merge,AveragePooling2D,concatenate
from keras.models import Model
from utils.BilinearUpSampling import *
from keras import optimizers


class SegModel(object):
        def __init__(self, input_size):
            self.input_size=input_size
            self._build_model()
            

        def relu(self,x):
            return Activation('relu')(x)

        def ResidualNet(self,nfilter,s):
            def Res_unit(x):
                BottleN = int(nfilter / 4)
                b_filter = BottleN

                x = BatchNormalization(axis=-1)(x)
                x = self.relu(x)
                ident_map = x

                x = Conv2D(b_filter,(1,1),strides=(s,s))(x)

                x = BatchNormalization(axis=-1)(x)
                x = self.relu(x)
                x = Conv2D(b_filter,3,3,border_mode='same')(x)
                x = BatchNormalization(axis=-1)(x)
                x = self.relu(x)
                x = Conv2D(nfilter,(1,1))(x)

                ident_map = Conv2D(nfilter,(1,1),strides=(s,s))(ident_map)

                out = merge([ident_map,x],mode='sum')

                return out
            return Res_unit

        def Res_Group(self,nfilter,layers,_stride):
            def Res_unit(x):
                for i in range(layers):
                    if i==0:
                        x = self.ResidualNet(nfilter,_stride)(x)
                    else:
                        x = self.ResidualNet(nfilter,1)(x)
                                   
                              
                return x
            return Res_unit

        #-------------------Pyramid Dilated Convolution--------------------------------
        def ASPP(self,input_layer):
            l = BatchNormalization(axis=-1)(input_layer)
            l = self.relu(l)
            conv1 = l
            dconv_filters=32
            a1 = Conv2D(dconv_filters, 1, activation = 'relu', padding = 'same', dilation_rate = 1)(conv1)
            a2 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 3)(conv1)
            a3 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 6)(conv1)
            a4 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 12)(conv1)
            a5 = Conv2D(dconv_filters, 3, activation = 'relu', padding = 'same', dilation_rate = 18)(conv1)

            concat = merge([a1,a2,a3,a4,a5], mode = 'concat', concat_axis = 3)  
            
            return concat


        def _build_model(self):
        #--------------------encoder---------    
            inp = Input(shape=(self.input_size))
            i = inp
            i = Conv2D(16,7,padding='same')(i)
        #----------------------------------------
            i = self.Res_Group(32,3,1)(i) 
        #----------------------------------------
            i = self.Res_Group(64,3,2)(i) 
        #---------------------------------------
            i = self.Res_Group(128,3,2)(i) 
            out_pdc2 = self.ASPP(i)
        #---------------------------------------

        #-----------------------decoder----------------
            i_dec = Dropout(0.15)(out_pdc2)

            conv_f = Conv2D(1,(1, 1), activation='sigmoid', padding='same')(i_dec)

            i_dec=BilinearUpSampling2D(size = (4,4))(conv_f)



            model = Model(inputs=inp, outputs=i_dec )

            self.model=model



