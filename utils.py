# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 19:08:14 2025

@author: sinam
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import sklearn.metrics



def _eval_perf(dataloader,backbone,fc, device):
    backbone.eval()
    fc.eval()
    pred=[]
    gt=[]
    for i, batch in enumerate(dataloader):
        xt, yt = batch["x"].to(device), batch["y"].to(device)
        outputs = backbone(xt)
        outputs=fc(outputs)


        yt_pred = torch.max(outputs, dim=1)[1]

        pred.extend(yt_pred.cpu().numpy())
        gt.extend(yt.cpu().numpy())

    F1s=sklearn.metrics.f1_score(np.array(gt),np.array(pred),average=None)
    acc = np.sum(np.array(pred)==np.array(gt)) / (len(dataloader.dataset))
    backbone.train()
    fc.train()
    return F1s, acc,np.sum(F1s)/3


def _eval_perf_withcount(dataloader,backbone,fc,device):
    backbone.eval()
    fc.eval()

    pred=[]
    gt=[]
    for i, batch in enumerate(dataloader):
        xt, yt = batch["x"].to(device), batch["y"].to(device)
        outputs = backbone(xt)
        outputs=fc(outputs)


        yt_pred = torch.max(outputs, dim=1)[1]

        pred.extend(yt_pred.cpu().numpy())
        gt.extend(yt.cpu().numpy())
    F1s=sklearn.metrics.f1_score(np.array(gt),np.array(pred),average=None)
    acc = np.sum(np.array(pred)==np.array(gt)) / (len(dataloader.dataset))
    backbone.train()
    fc.train()
    return F1s, acc,np.sum(F1s)/3,np.unique(np.array(pred),return_counts=True)[1]






def mean_confidence_score(dataloader,backbone,fc,device):
    backbone.eval()
    fc.eval()
    all_output_soft = []
    with torch.no_grad():

        for batch in dataloader:
            xt, _ = batch["x"].to(device), batch["y"].to(device)
            outputs = backbone(xt)
            outputs=fc(outputs)


            yt_pred = F.softmax(outputs, dim=1)
            all_output_soft.append(yt_pred.float().cpu())

        all_output_soft = torch.cat(all_output_soft, dim=0)
        max_probs, _ = torch.max(all_output_soft, dim=-1)



        backbone.train()
        fc.train()
        return np.mean(max_probs.cpu().numpy())




def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer




class CropMappingDataset(Dataset):
    """
    crop classification dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"sample_x": self.x[idx], "sample_y": self.y[idx]}

def _collate_fn(batch):
    """
    define how to aggregate samples to batch
    """
    return {
        "x": torch.FloatTensor(
            np.array([sample["sample_x"] for sample in batch])
        ),
        "y": torch.LongTensor(
            np.array([sample["sample_y"] for sample in batch])
        )
    }