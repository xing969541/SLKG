# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd

with open("PredData_gg.pkl", "br") as file:
    (src, dst), mat = pickle.load(file)
    src = src.detach().cpu().numpy()
    dst = dst.detach().cpu().numpy()
uv = np.load("uvdata_gg.npy")

ggMat = mat*uv

with open("embidDict.pkl", "br") as file:
    di = pickle.load(file)
    gdi = di["gene"]
    igdi = {v:k for k,v in gdi.items()}

pred = ggMat.copy()
pred[src,dst] = 0
threshold = 0.254
l, r = np.where(pred>threshold)
tri = (pred[l,r], [igdi[x] for x in l], [igdi[x] for x in r])
tri = list(zip(*tri))
tri.sort(key=lambda x:x[0], reverse=True)

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
        
count = 0
for s, l, r in tri:
    if l in sl and r in sl[l]:
        if s>threshold:
#            print(','.join([str(x) for x in table.iloc[sl[l][r]]]+[str(round(s,4))]))
#        print(s,l,r)
            count+=1
print(count)