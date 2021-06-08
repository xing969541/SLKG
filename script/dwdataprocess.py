# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score, roc_curve

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def cal(pos, neg):
    scores = torch.reshape(torch.cat([pos, neg]), (-1,))
    labels = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])])
    p, r, t = precision_recall_curve(labels, scores)
    AUPR = auc(r, p)
    AUROC = roc_auc_score(labels, scores)
    F1 = max(f1_score(labels, scores>x) for x in np.arange(0.01,1,0.01))
    return AUPR, AUROC, F1

def func1(tuple_):
    return ' '.join("{:.4f} {:.4f}".format(m,s) for m, s in zip(*tuple_))

def anova(rawData):
    df = pd.DataFrame([[k, *data] for k, v in rawData.items() for data in v])
    df.columns = ["Class", "time", "AUPR", "AUC", "F1"]
    for k in df.columns[1:]:
        anovat = anova_lm(ols('{} ~ Class'.format(k), df).fit())
        print(k, anovat, sep='\n', end='\n\n')

def main():
    dataDir = "./dwScoreData"
    rawData = {}
    for fileName in os.listdir(dataDir):
#        if not fileName[1] in ['r', 'l']: continue
        if not('nl' in fileName or 'lr2' in fileName): continue
        tag = fileName.split('_')[0]
#        tag = tuple(fileName.split('_')[:-1])
        with open(os.path.join(dataDir, fileName), 'br') as file:
            (_, time), (pos, neg) = pickle.load(file)
        pos = pos.cpu()
        neg = neg.cpu()
        if tag in rawData:
            rawData[tag].append((time, *cal(pos, neg)))
        else:
            rawData[tag] = [(time, *cal(pos, neg))]
    for k, v in rawData.items():
        rawData[k] = np.array(v)
    anova(rawData)
    data = {k:(v.mean(0), v.std(0)) for k, v in rawData.items()}
    merged = {k:func1(v) for k,v in data.items()}
    print("\ttime\t\tAUPR\t\tAUROC\t\tF1")
    for k,v in merged.items():
        print(k, v)
    return data, rawData, merged

if __name__ == "__main__":
    data, rawData, merged = main()