# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

def baseCal(pos, neg, threshold):
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

def cal(pos, neg):
    data = np.array([baseCal(pos, neg, i) for i in helper(1, 0.01, 0.01)])
    return data

def calAucF1(data):
    F1 = data[:,2].max()
    return calArea(data[:,0], data[:,1]), F1, int(np.where(data[:,2] == F1)[0].mean())*0.01


def func1(tuple_):
    return '\t'.join("{:.4f}\t{:.4f}".format(m,s) for m, s in zip(*tuple_))

def main():
    rawData = {}
    for fileName in [x for x in os.listdir() if x.startswith("data")]:
        tag = fileName.split('_')[1]
        with open(fileName, 'br') as file:
            pos, neg = pickle.load(file)
        rawData[tag] = cal(pos, neg)
        
    figure, axes = plt.subplots(3, 2, figsize=(18,12), dpi=800)
    for i, (k, v) in enumerate(rawData.items()):
        axes[i][0].plot([0]+list(v[:,0])+[1], [0]+list(v[:,1])+[1])
        axes[i][0].set_title("{} ROC".format(k))
        axes[i][1].plot(list(helper(1, 0, 0.01)), list(v[:,2])+[0])
        axes[i][1].set_title("{} F1".format(k))
    plt.savefig("./fig.png")
    plt.show()
    
    print("tag\tauc\tmax F1\tmax F1 threshold\t")
    print('\n'.join('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format(k, *calAucF1(v)) for k,v in rawData.items()))
    return rawData

if __name__ == "__main__":
    rawData = main()