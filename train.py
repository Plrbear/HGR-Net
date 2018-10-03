
import os
import sys
from keras.callbacks import *
from keras.models import load_model
from keras import optimizers
from Rec_model import RecModel
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import argparse

       
def train(opt):  

    input_size=(opt.row,opt.col,opt.ch)

    #Data Generator

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
         )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        opt.train_ad,  # this is the target directory
        target_size=(opt.row, opt.col),  # all images will be resized to 150x150
        batch_size=opt.batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            opt.validation_ad,
            target_size=(opt.row, opt.col),
            batch_size=opt.batch_size,
            class_mode='categorical')

    # check the name of each class with corresponding indices using:
    # train_generator.class_indices


    
    #####Compile####
    RecM=RecModel(input_size,opt.num_class)
    model=RecM.model_F
    _adam=optimizers.Adam(lr=opt.lr, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(loss='binary_crossentropy',optimizer = _adam,metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint(opt.chekp, monitor='val_acc',verbose=1, save_best_only=True)

    model.fit_generator(
            train_generator,
            steps_per_epoch=opt.num_img // opt.batch_size,
            epochs=opt.epochs,
            validation_data=validation_generator,
            validation_steps=opt.num_val // opt.batch_size,
            callbacks=[model_checkpoint])


            
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=2)
    # parser.add_argument('--input_size', type=list, default='(320,320,3)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--train_ad', type=str, default='')
    parser.add_argument('--validation_ad', type=str, default='')

    parser.add_argument('--chekp', type=str, default='')
    parser.add_argument('--row', type=int, default=320)
    parser.add_argument('--col', type=int, default=320)
    parser.add_argument('--ch', type=int, default=3)
    parser.add_argument('--num_val', type=int, default=3)
    parser.add_argument('--num_img', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=3)



    opt = parser.parse_args()
    train(opt)            
