#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 01:12:02 2020

@author: kuangmeng
"""

from keras.layers import Concatenate, LeakyReLU, Conv3D, UpSampling3D, Input, BatchNormalization, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import os
from keras.models import load_model

class UNet3D():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.structure()
        self.model.compile(optimizer = Adam(lr = 1e-4), 
                           loss = 'binary_crossentropy', 
                           metrics = ['accuracy'])
        
    def structure(self):
        inputs = Input(self.input_shape)
        
        conv1 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 32)(inputs)
        conv1 = BatchNormalization()(conv1)
        meg1 = LeakyReLU()(conv1)
        conv1 = MaxPooling3D(pool_size=(1, 2, 2))(meg1)
        
        conv2 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 128)(conv1)
        conv2 = BatchNormalization()(conv2)
        meg2 = LeakyReLU()(conv2)
        conv2 = MaxPooling3D(pool_size=(1, 2, 2))(meg2)
        
        conv3 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 512)(conv2)
        conv3 = BatchNormalization()(conv3)
        meg3 = LeakyReLU()(conv3)
        conv3 = MaxPooling3D(pool_size=(1, 2, 2))(meg3)
        
        conv4 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 1024)(conv3)
        conv4 = BatchNormalization()(conv4)
        meg4 = LeakyReLU()(conv4)
        conv4 = MaxPooling3D(pool_size=(1, 2, 2))(meg4)

        conv5 = Conv3D(kernel_size = (1, 1, 1), padding = 'same', filters = 1024)(conv4)
        conv5 = BatchNormalization()(conv5)
        conv5 = LeakyReLU()(conv5)
        
        
        up1 = UpSampling3D(size = (1, 2, 2))(conv5)
        up1 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 512)(up1)
        up1 = BatchNormalization()(up1)
        up1 = LeakyReLU()(up1)
        up1 = Concatenate(axis = -1)([meg4,up1])

        up2 = UpSampling3D(size = (1, 2, 2))(up1)
        up2 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 256)(up2)
        up2 = BatchNormalization()(up2)
        up2 = LeakyReLU()(up2)
        up2 = Concatenate(axis = -1)([meg3,up2])

        
        up3 = UpSampling3D(size = (1, 2, 2))(up2)
        up3 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 128)(up3)
        up3 = BatchNormalization()(up3)
        up3 = LeakyReLU()(up3)
        up3 = Concatenate(axis = -1)([meg2, up3])

        
        up4 = UpSampling3D(size = (1, 2, 2))(up3)
        up4 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 32)(up4)
        up4 = BatchNormalization()(up4)
        up4 = LeakyReLU()(up4)
        up4 = Concatenate(axis = -1)([meg1, up4])

        up5 = Conv3D(kernel_size = (3, 3, 3), padding = 'same', filters = 1)(up4)
        outputs = LeakyReLU()(up5)
        
        model = Model(inputs, outputs)
        
        model.summary()
        
        return model

    def train(self, train_set, epochs = 100000, batch_size = 4, save_interval = 100, mode = 'ED'):
        X_train = train_set[mode]
        Y_train = train_set[mode + '_GT']
        for epoch in range(epochs):
            split = np.random.randint(0, X_train.shape[0], batch_size)
            x_train_batch = np.resize(X_train[split], (batch_size, X_train[0].shape[0], X_train[0].shape[1], X_train[0].shape[2], 1))
            y_train_batch = np.resize(Y_train[split], (batch_size, Y_train[0].shape[0], Y_train[0].shape[1], Y_train[0].shape[2], 1))
            loss = self.model.train_on_batch(x_train_batch, y_train_batch)
            print('%d [loss: %f, acc: %f]'%(epoch, loss[0], loss[1]))
            if epoch % save_interval == 0:
                if not os.path.exists("saved_models"):
                    os.makedirs("saved_models")
                self.model.save("saved_models/epoch%06d.h5" % epoch)
    
    def evaluate(self, test_result, test_gt):
        return 
    
    def test(self, test_set, model_path, evaluate = None, test_gt = None, mode = 'ED'):
        self.model = load_model(model_path)
        test_set = np.resize(test_set, (test_set.shape[0], test_set[0].shape[1], test_set[0].shape[2], test_set[0].shape[3], 1))
        test_result = self.model.predict(test_set)
        if evaluate != None:
            self.evaluate(test_result, test_gt)
        
        return test_result
        

if __name__ == "__main__":
    unet_3d = UNet3D((10, 256, 256, 1))
    from load_data import LoadData
    npy_dir = './processed_ACDC'
    ld = LoadData(npy_dir)
    ld.load_data_dict()
    train, test, _ = ld.data_split()
    unet_3d.train(train)
        
        
        
        
        
    
        