# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/Inshop.py
from .base import *

import numpy as np, os, sys, pandas as pd, csv, copy
import torch
import torchvision
import PIL.Image


class Inshop_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/Inshop_Clothes'
        self.mode = mode
        self.transform = transform
        self.train_ys, self.train_im_paths = [], []
        self.query_ys, self.query_im_paths = [], []
        self.gallery_ys, self.gallery_im_paths = [], []
                    
        data_info = np.array(pd.read_table(self.root +'/Eval/list_eval_partition.txt', header=1, delim_whitespace=True))[:,:]
        #Separate into training dataset and query/gallery dataset for testing.
        train, query, gallery = data_info[data_info[:,2]=='train'][:,:2], data_info[data_info[:,2]=='query'][:,:2], data_info[data_info[:,2]=='gallery'][:,:2]

        #Generate conversions
        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in train[:,1]])))}
        train[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in train[:,1]])

        lab_conv = {x:i for i,x in enumerate(np.unique(np.array([int(x.split('_')[-1]) for x in np.concatenate([query[:,1], gallery[:,1]])])))}
        query[:,1]   = np.array([lab_conv[int(x.split('_')[-1])] for x in query[:,1]])
        gallery[:,1] = np.array([lab_conv[int(x.split('_')[-1])] for x in gallery[:,1]])

        #Generate Image-Dicts for training, query and gallery of shape {class_idx:[list of paths to images belong to this class] ...}
        for img_path, key in train:
            self.train_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.train_ys += [int(key)]

        for img_path, key in query:
            self.query_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.query_ys += [int(key)]

        for img_path, key in gallery:
            self.gallery_im_paths.append(os.path.join(self.root, 'Img', img_path))
            self.gallery_ys += [int(key)]
            
        if self.mode == 'train':
            self.im_paths = self.train_im_paths
            self.ys = self.train_ys
        elif self.mode == 'query':
            self.im_paths = self.query_im_paths
            self.ys = self.query_ys
        elif self.mode == 'gallery':
            self.im_paths = self.gallery_im_paths
            self.ys = self.gallery_ys

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]

        return im, target