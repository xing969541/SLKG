# -*- coding: utf-8 -*-

from uv import main
import pickle
import numpy as np
import pandas as pd


with open("embidDict.pkl", "br") as file:
    di = pickle.load(file)
    gdi = di["gene"]
    igdi = {v:k for k,v in gdi.items()}
with open("PredData_gg.pkl", "br") as file:
    (src, dst), mat = pickle.load(file)
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()

table = pd.read_csv("Human_SL.csv", header=0)
sl = {}
for i, (l,r) in enumerate(zip(table["gene_a.name"], table["gene_b.name"])):
    if l in sl:
        sl[l][r] = i
    else:
        sl[l] = {r:i}
    if r in sl:
        sl[r][l] = i
    else:
        sl[r] = {l:i}
        
save = 1000
for i in range(200):
    main('gg')
    uv = np.load("uvdata_gg.npy")
    
    pred = mat*uv
    pred[src,dst] = 0
    
    l, r = np.where(pred>0.3)
    tri = (pred[l,r], [igdi[x] for x in l], [igdi[x] for x in r])
    tri = list(zip(*tri))
    tri.sort(key=lambda x:x[0], reverse=True)
    count = sum(1 for s, l, r in tri if l in sl and r in sl[l])
    tmp = len(l)/count
    print(i, tmp)
    if tmp < save:
        save = tmp
        with open('bestgguv.pkl','bw') as file:
            pickle.dump(uv,file)
    