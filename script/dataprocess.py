# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

def cal(pos, neg, threshold):
    TP = (torch.sigmoid(pos)>threshold).sum().item()+1e-10
    FN = (torch.sigmoid(pos)<threshold).sum().item()+1e-10
    TN = (torch.sigmoid(neg)<threshold).sum().item()+1e-10
    FP = (torch.sigmoid(neg)>threshold).sum().item()+1e-10
    
    p = TP / (TP + FP)
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    F1 = 2 * TPR * p / (TPR + p)
    return FPR, TPR, F1

def calArea(x, y):
    x = [0] + list(x) + [1]
    y = [0] + list(y) + [1]
    return sum((y[i]+y[i+1])*(x[i+1]-x[i]) for i in range(len(x)-1)) / 2

def helper(l, r, s):
    while l > r:
        yield l
        l -= s

def calAucF1(pos, neg):
    data = np.array([cal(pos, neg, i) for i in helper(1, 0.01, 0.01)])
    return calArea(data[:,0], data[:,1]), data[:,2].max()

def func1(tuple_):
    return '\t'.join("{:.4f}±{:.4f}".format(m,s) for m, s in zip(*tuple_))
    
def main():
    dataDir = "./scoreData"
    rawData = {}
    for fileName in os.listdir(dataDir):
        tag = fileName.split('_')
        tag = tag[0] if len(tag)==2 else tuple(tag[:2])
        with open(os.path.join(dataDir, fileName), 'br') as file:
            (_, time), (pos, neg) = pickle.load(file)
        if tag in rawData:
            rawData[tag].append((time, *calAucF1(pos, neg)))
        else:
            rawData[tag] = [(time, *calAucF1(pos, neg))]
    for k,v in rawData.items():
        rawData[k] = np.array(v)
    data = {k:(v.mean(0),v.std(0)) for k,v in rawData.items()}
    merged1 = {k:func1(v) for k,v in data.items()}
    merged2 = {}
    for k, v in merged1.items():
        if type(k) == str:
            merged2[k] = v
        elif k[0] in merged2:
            merged2[k[0]][k[1]] = v
        else:
            merged2[k[0]] = {k[1]:v}
    merged3 = {}
    for  k,v in merged2.items():
        if type(v)==str:
            merged3[k] = v
        else:
            merged3[k] = '\t'.join([v["dg"], v["gg"], v["gc"]])
    print('\n'.join(k+'\t'+v for k,v in merged3.items()))
    return data, rawData, merged3

if __name__ == "__main__":
    data, rawData, merged = main()