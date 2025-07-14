# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:28:19 2025

@author: sinam
"""


import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics
import argparse
import os
import re
import glob
from models import cnn, PETransformerModel, DCM, FC
from utils import _eval_perf_withcount, mean_confidence_score, op_copy, CropMappingDataset,_collate_fn



np.random.seed(10)
torch.manual_seed(10)



def args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./Data_USA/')
    parser.add_argument("--source_site", type=str, choices=['A', 'B','C'])
    parser.add_argument("--source_year", type=str, choices=['2019', '2020','2021'])
    parser.add_argument("--target_site", type=str, choices=['A', 'B','C'])
    parser.add_argument("--target_year", type=str, choices=['2019', '2020','2021'])
    parser.add_argument("--pretrained_save_dir", type=str, default='./Pretrained_USA/')
    parser.add_argument("--adapted_save_dir", type=str, default='./Adapted/')
    parser.add_argument("--backbone_network", type=str, choices=['CNN', 'LSTM','Transformer'])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=0.001)
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












    image_target=np.load(cfg.data_dir+"/Site_"+cfg.target_site+"/x-"+cfg.target_year+".npy")
    image_target=image_target*(0.0000275) -0.2
    image_target= ( image_target-np.mean(image_target,axis=(0,1),keepdims=True) )/ np.std(image_target,axis=(0,1),keepdims=True)
    if cfg.backbone_network=="CNN":
        image_target=np.transpose(image_target, (0,2,1))
    label_target=np.load(cfg.data_dir+"/Site_"+cfg.target_site+"/y-"+cfg.target_year+".npy")

    TargetLoader=DataLoader(CropMappingDataset(image_target, label_target),batch_size=cfg.batch_size, shuffle=True,num_workers=0,collate_fn=_collate_fn)








    param_group = []
    param_group_c = []

    lr=cfg.learning_rate

    for k, v in backbone.named_parameters():
        
        if True:
            param_group += [{"params": v, "lr": lr * 0.1}]  


    for k, v in fc.named_parameters():
        param_group_c += [{"params": v, "lr": lr * 1}]  





    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)



    backbone.load_state_dict(torch.load(cfg.pretrained_save_dir+'/backbone'+'Site'+cfg.source_site+cfg.source_year+'.pth'))
    fc.load_state_dict(torch.load(cfg.pretrained_save_dir+'/fc'+'Site'+cfg.source_site+cfg.source_year+'.pth'))

    F1s, acc,mF1s,counts = _eval_perf_withcount(TargetLoader,backbone,fc,device)

    print("BeforeAdpatation=>","acc:",acc, "- F1s:",F1s,"- mF1s:",mF1s,"- counts:",counts)



    for beta in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        backbone.load_state_dict(torch.load(cfg.pretrained_save_dir+'/backbone'+'Site'+cfg.source_site+cfg.source_year+'.pth'))
        fc.load_state_dict(torch.load(cfg.pretrained_save_dir+'/fc'+'Site'+cfg.source_site+cfg.source_year+'.pth'))
        backbone.eval()
        fc.train()
        losses = 0
        correct = 0
        for i, batch in enumerate(TargetLoader):
            xt, yt = batch["x"].to(device), batch["y"].to(device)
            outputs = backbone(xt)
            outputs=fc(outputs)
            alpha=5
            q_probs = outputs.softmax(1)
            q_ent = ((1 - (torch.pow(q_probs.mean(0), alpha)).sum(0)) / (alpha - 1))
            q_cond_ent = ((1 - (torch.pow(q_probs + 1e-12, alpha)).sum(1)) / (alpha - 1)).mean(0)
            loss1 = -beta*q_ent + q_cond_ent

            optimizer.zero_grad()
            optimizer_c.zero_grad()
            loss1.backward()
            optimizer.step()
            optimizer_c.step()


        F1s,acc,mF1s,counts = _eval_perf_withcount(TargetLoader,backbone,fc,device)

        print("AdpatationWithBeta="+str(beta)+"=>","acc:",acc, "- F1s:",F1s,"- mF1s:",mF1s,"- counts:",counts)

        if len(counts)<3:
            continue
        else:

            if counts[0]/len(label_target)< 0.05 or counts[1]/len(label_target)<  0.05 or counts[2]/len(label_target)<  0.05:
                continue

        mean_confidence=mean_confidence_score(TargetLoader,backbone,fc,device)
        print("mean_confidence_score:",mean_confidence)
        if not os.path.exists(cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year):
            os.makedirs(cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year)
        torch.save(backbone.state_dict(), cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'/backbone'+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'conf'+str(mean_confidence)+'.pth')
        torch.save(fc.state_dict(), cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'/fc'+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'conf'+str(mean_confidence)+'.pth')






    for gamma in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        backbone.load_state_dict(torch.load(cfg.pretrained_save_dir+'/backbone'+'Site'+cfg.source_site+cfg.source_year+'.pth'))
        fc.load_state_dict(torch.load(cfg.pretrained_save_dir+'/fc'+'Site'+cfg.source_site+cfg.source_year+'.pth'))
        backbone.eval()
        fc.train()
        losses = 0
        correct = 0
        for i, batch in enumerate(TargetLoader):
            xt, yt = batch["x"].to(device), batch["y"].to(device)
            outputs = backbone(xt)
            outputs=fc(outputs)
            alpha=5
            q_probs = outputs.softmax(1)
            q_ent = ((1 - (torch.pow(q_probs.mean(0), alpha)).sum(0)) / (alpha - 1))
            q_cond_ent = ((1 - (torch.pow(q_probs + 1e-12, alpha)).sum(1)) / (alpha - 1)).mean(0)
            loss1 = -gamma*q_ent + q_cond_ent

            optimizer.zero_grad()
            optimizer_c.zero_grad()
            loss1.backward()
            optimizer.step()
            optimizer_c.step()
 
        F1s,acc,mF1s,counts = _eval_perf_withcount(TargetLoader,backbone,fc,device)

        print("AdpatationWithGamma="+str(gamma)+"=>","acc:",acc, "- F1s:",F1s,"- mF1s:",mF1s,"- counts:",counts)

        if len(counts)<3:
            continue
        else:

            if counts[0]/len(label_target)< 0.05 or counts[1]/len(label_target)<  0.05 or counts[2]/len(label_target)<  0.05:
                continue

        mean_confidence=mean_confidence_score(TargetLoader,backbone,fc,device)
        print("mean_confidence_score:",mean_confidence)
        if not os.path.exists(cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year):
            os.makedirs(cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year)
        torch.save(backbone.state_dict(), cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'/backbone'+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'conf'+str(mean_confidence)+'.pth')
        torch.save(fc.state_dict(), cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'/fc'+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'conf'+str(mean_confidence)+'.pth')














    ####### Now that the adaptation is complete, we evaluate the performance using the code below:

    TargetLoader=DataLoader(CropMappingDataset(image_target, label_target),batch_size=cfg.batch_size, shuffle=False,num_workers=0,collate_fn=_collate_fn)

    paths_unsorted=glob.glob(cfg.adapted_save_dir+'/'+cfg.backbone_network+"-"+cfg.source_site+cfg.source_year+'to'+cfg.target_site+cfg.target_year+'/*.pth')
    paths = sorted(paths_unsorted)
    probs=[]
    for n,j in enumerate(paths[0:int(len(paths_unsorted)/2)]):
        match = re.search(r"conf(\d+\.\d+)", j)

        if match:
            extracted_value = match.group(1)
            probs.append(extracted_value)

    backbones=[]
    fcs=[]

    all_preds=[]

    for i in range(len(paths[0:int(len(paths_unsorted)/2)])):

        if cfg.backbone_network=="CNN":
            backbone=cnn()
            fc=FC(input_dim=1024)
        elif cfg.backbone_network=="Transformer":
            backbone=PETransformerModel()
            fc=FC(input_dim=64)
        elif cfg.backbone_network=="LSTM":
            backbone=DCM()
            fc=FC(input_dim=512)



        backbone = backbone.to(device)
        fc = fc.to(device)
        backbone = torch.nn.DataParallel(backbone, device_ids=cfg.gpu)
        fc = torch.nn.DataParallel(fc, device_ids=cfg.gpu)

        backbones.append(backbone)
        fcs.append(fc)


    all_preds=[]
    all_weights=[]
    for n,j in enumerate(range(len(paths[0:int(len(paths_unsorted)/2)]))):


            backbones[n].load_state_dict(torch.load(paths[j]))
            fcs[n].load_state_dict(torch.load(paths[j+int(len(paths_unsorted)/2)]))
            all_output_soft=[]
            for _,batch in enumerate(TargetLoader):
                backbones[n].eval()
                fcs[n].eval()
                xt, yt = batch["x"].to(device), batch["y"].to(device)
                outputs = backbones[n](xt)
                outputs=fcs[n](outputs)

                yt_pred = F.softmax(outputs, dim=1)
                all_output_soft.extend(yt_pred.float().cpu().detach().numpy())

            new_max=1
            new_min=0.1
            old_min=float(min(probs))
            old_max=float(max(probs))
            weight= ((float(probs[n]) - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            if weight==1:
                pred_maxprob=np.array(all_output_soft)
            all_preds.append(((weight))*np.array(all_output_soft))
            all_weights.append((weight))



    prs=np.sum(np.array(all_preds),axis=0)/np.sum(np.array(all_weights))


    print("macro",sklearn.metrics.f1_score(label_target,np.argmax(prs,-1),average='macro'))
    print("weighted",sklearn.metrics.f1_score(label_target,np.argmax(prs,-1),average='weighted'))
    print("micro",sklearn.metrics.f1_score(label_target,np.argmax(prs,-1),average='micro'))

    np.save("dict"+cfg.source_site+cfg.source_year+cfg.target_site+cfg.target_year+".npy",{"macro":sklearn.metrics.f1_score(label_target,np.argmax(prs,-1),average='macro'),
                                              "weighted":sklearn.metrics.f1_score(label_target,np.argmax(prs,-1),average='weighted'),
                                              "micro":sklearn.metrics.f1_score(label_target,np.argmax(prs,-1),average='micro')})


