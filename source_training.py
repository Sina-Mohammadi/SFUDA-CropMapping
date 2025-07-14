# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 19:30:29 2025

@author: sinam
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os
from torch.utils.data import DataLoader
from models import cnn, PETransformerModel, DCM, FC
from utils import _eval_perf, CropMappingDataset,_collate_fn



np.random.seed(10)
torch.manual_seed(10)



def args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='Data_USA')
    parser.add_argument("--source_site", type=str, choices=['A', 'B','C'])
    parser.add_argument("--source_year", type=str, choices=['2019', '2020','2021'])
    parser.add_argument("--pretrained_save_dir", type=str, default='Pretrained_USA')
    parser.add_argument("--backbone_network", type=str, choices=['CNN', 'LSTM','Transformer'])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--gpu", type=list, default=[0])

    return parser.parse_args()







if __name__ == "__main__":


    cfg = args()
    if cfg.backbone_network=="CNN":
        backbone=cnn()
        fc=FC(input_dim=1024)
    elif cfg.backbone_network=="Transformer":
        backbone=PETransformerModel()
        fc=FC(input_dim=64)
    elif cfg.backbone_network=="LSTM":
        backbone=DCM()
        fc=FC(input_dim=512)
    device = torch.device(f'cuda:{cfg.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    fc = fc.to(device)

    backbone = torch.nn.DataParallel(backbone, device_ids=cfg.gpu)
    fc = torch.nn.DataParallel(fc, device_ids=cfg.gpu)
    total_params = sum(p.numel() for p in backbone.parameters())+sum(p.numel() for p in fc.parameters())
    print("Total number of parameters: ", total_params)


    image=np.load(cfg.data_dir+"/Site_"+cfg.source_site+"/x-"+cfg.source_year+".npy")
    image=image*(0.0000275) -0.2
    image= ( image-np.mean(image,axis=(0,1),keepdims=True) )/ np.std(image,axis=(0,1),keepdims=True)
    if cfg.backbone_network=="CNN":
        image=np.transpose(image, (0,2,1))
    label=np.load(cfg.data_dir+"/Site_"+cfg.source_site+"/y-"+cfg.source_year+".npy")



    random.seed(10)


    val_indices=np.random.choice(np.arange(len(label)), size=int(len(label)/5), replace=False)

    np.save("val_indices_"+cfg.source_site+cfg.source_year+".npy",val_indices)


    j = np.arange(len(label))
    train_indices = np.delete(j,val_indices)


    images_train = image[train_indices]
    labels_train = label[train_indices]

    images_val = image[val_indices]
    labels_val = label[val_indices]
    trainLoader=DataLoader(CropMappingDataset(images_train, labels_train),batch_size=cfg.batch_size, shuffle=True,num_workers=0,collate_fn=_collate_fn)
    val_laoder=DataLoader(CropMappingDataset(images_val, labels_val),batch_size=cfg.batch_size, shuffle=True,num_workers=0,collate_fn=_collate_fn)


    optimizer = optim.Adam(list(backbone.parameters())+list(fc.parameters()), lr=0.0001)

    criterion= nn.CrossEntropyLoss()





    best_mF1s=0
    for epoch in range(1, 1000):

        backbone.train()
        fc.train()
        for i, batch in enumerate(trainLoader):
            xt_train_batch = batch["x"].to(device)
            yt_train_batch = batch["y"].to(device)
            optimizer.zero_grad()
            outputs = backbone(xt_train_batch)
            outputs=fc(outputs)

            loss = criterion(outputs, yt_train_batch)
            loss.backward()
            optimizer.step()




        _,acc_train,_ = _eval_perf(trainLoader,backbone,fc,device)

        F1s, acc,mF1s = _eval_perf(val_laoder,backbone,fc,device)


        if mF1s>best_mF1s:
            best_mF1s=mF1s
            if not os.path.exists(cfg.pretrained_save_dir):
                os.makedirs(cfg.pretrained_save_dir)
            torch.save(backbone.state_dict(), cfg.pretrained_save_dir+'/backbone'+'Site'+cfg.source_site+cfg.source_year+'.pth')
            torch.save(fc.state_dict(), cfg.pretrained_save_dir+'/fc'+'Site'+cfg.source_site+cfg.source_year+'.pth')






        print(epoch,"acc_train:",acc_train)

        print(epoch,"mF1s_val:",mF1s)
