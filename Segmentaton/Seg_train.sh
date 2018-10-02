#!/bin/bash

python train.py --batch_size 2 --epochs 200 --lr 0.001 \
                --ad1 '/home/amir/dataset1/fold1/' \
                --ad2 'train' \
                --ad3 'validation' \
                --ad5 'masks' \
		--img_format '*png' --chekp 'Seg_weight' \
                --row 320 --col 320 --ch 3  

                 

