#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:42:15 2020

@author: kuangmeng
"""
import os
import h5py
import numpy as np

class LoadH5():
    def __init__(self, data_dir, save_path):
        self.data_dir = data_dir
        self.data = {'data':[], 'label': []}
        self.save_file = data_dir.split('/')[-1] + '.npy' if len(data_dir.split('/')[-1]) >= 1 else data_dir.split('/')[-2] + '.npy'
        self.save_path = save_path
        
    
    def readSingleH5(self, item_path):
        f = h5py.File(item_path, 'r')
        # keyset = f.keys() #get set of cine slice names enclosed in the file
        # keylist = list(f) #get list of cine slice names enclosed in the file
        key = 'MID' #define the name of the cine slice requested
        img = f[key]['image'][:] #get the 50*512*512*4 image array data as img
        # the label data will be a float array. You may want to normalise the data before use
        lbl = f[key]['label'][:] #get the 50*512*512 label array data as lbl
        # the label data will be a binary (0/1) float array
        f.close()
        return img, lbl
    
    def read(self):
        dir_list = os.listdir(self.data_dir)
        for item_path in dir_list:
            if os.path.exists(os.path.join(self.data_dir, item_path)) and '.h5' in item_path:
                img, lbl = self.readSingleH5(os.path.join(self.data_dir, item_path))
                self.data['data'].append(img)
                self.data['label'].append(lbl)
        return self.data
    
    def save(self):
        np.save(os.path.join(self.save_path, self.save_file), self.data)
    
if __name__ == '__main__':
    lh5 = LoadH5('/Users/kuangmeng/Documents/SpyderProjects/mengutils/mengutils', './')
    data = lh5.read()
    print(data['data'][0].shape)
    lh5.save()
    
    
    
    
    
    
    
    
    
    