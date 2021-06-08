# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score, roc_curve
plt.rcParams['figure.dpi'] = 400

def round_(arr):
    return np.array(np.array(arr*1000, dtype=np.int32), dtype=arr.dtype) / 1000

def cal(src, dst, mat, pmat, info):
    labels = np.zeros(pmat.shape)
    labels[src, dst] = 1
    labels = labels.flatten()
#    combineMat = pmat.flatten()
    combineMat = (pmat * mat).flatten()
#    combineMat = round_(combineMat)
    
    p, r, t = precision_recall_curve(labels, combineMat)
    aupr = auc(r, p)
    auroc = roc_auc_score(labels, combineMat)
    F1li = [f1_score(labels, combineMat>x) for x in np.arange(0.01, 1, 0.01)]
    print(info)
    print("AUPR\tAUC\tF1\tmax F1 threshold")
    print(*(round(x,4) for x in [aupr, auroc, max(F1li)]), F1li.index(max(F1li))*0.01, sep='  ')
    
    plt.plot(np.arange(0.01, 1, 0.01), F1li)
    plt.xlabel('threshold')
    plt.ylabel('F1')
    plt.ylim((0, 1))
    plt.savefig("F1 curve_{}.png".format(info.replace('\t', '_')))
    plt.cla()
    
    plt.plot(r, p)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig("PRC_{}.png".format(info.replace('\t', '_')))
    plt.cla()
    print()

with open("PredData_gg.pkl", "br") as file:
    (src, dst), mat = pickle.load(file)
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()
    
#cal(src, dst, mat, np.load("wdata_gg.npy"), "g-g\tw")
cal(src, dst, mat, np.load("uvdata_gg.npy"), "g-g\tuv")

with open("PredData_gc.pkl", "br") as file:
    (src, dst), mat = pickle.load(file)
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()

cal(src, dst, mat, np.load("uvdata_gc.npy"), "g-c\tuv")


with open("PredData_dg.pkl", "br") as file:
    (src, dst), mat = pickle.load(file)
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()

cal(src, dst, mat, np.load("uvdata_dg.npy"), "d-g\tuv")

