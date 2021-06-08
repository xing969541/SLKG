# -*- coding: utf-8 -*-

import pickle
import numpy as np

overCount = lambda arr, threshold: len(np.where(arr>threshold)[0])

def load(matFileName, uvFileName):
    with open(matFileName, "br") as file:
        (src, dst), mat = pickle.load(file)
    uv = np.load(uvFileName)
#    print(overCount(mat, 0.5))
    return mat * uv

def mat2di(mat, threshold):
    di = {}
    if mat.shape[0] != mat.shape[1]:
        for l, r in zip(*np.where(mat > threshold)):
            if l in di:
                di[l][r] = mat[l,r]
            else:
                di[l] = {r:mat[l,r]}
    else:
        for l, r in zip(*np.where(mat > threshold)):
            if l in di:
                di[l][r] = mat[l,r]
            else:
                di[l] = {r:mat[l,r]}
            if r in di:
                di[r][l] = mat[l,r]
            else:
                di[r] = {l:mat[l,r]}
    return di

def norm(mat, threshold):
    mat[mat<threshold] = threshold
    mat = 1 - (1-mat)*(1/(1-threshold))
    return mat

with open("embidDict.pkl", "br") as file:
    di = pickle.load(file)
    gdi = di["gene"]
    igdi = {v:k for k,v in gdi.items()}
    ddi = di["drug"]
    iddi = {v:k for k,v in ddi.items()}
    cdi = di["cancer"]
    icdi = {v:k for k,v in cdi.items()}

ggMat = load("PredData_gg.pkl", "uvdata_gg.npy")
gcMat = load("PredData_gc.pkl", "uvdata_gc.npy")

with open("PredData_dg.pkl", "br") as file:
    (src, dst), dgMat = pickle.load(file)
#    print(overCount(dgMat, 0.5))
    dgMat = norm(dgMat, 0.994)
    

gg = mat2di(ggMat, 0.18)
gc = mat2di(gcMat.T, 0.16)
dg = mat2di(dgMat, 0)

#threshold = 0.1
#res = []
#for k1,v1 in dg.items():
#    for k2,s1 in v1.items():
#        if k2 in gg:
#            for k3, s2 in gg[k2].items():
#                if k3 in gc:
#                    for k4, s3 in gc[k3].items():
#                        score = s1*s2*s3
#                        if score > threshold:
#                            res.append([iddi[k1],igdi[k2],igdi[k3],icdi[k4],score])
#res.sort(reverse=True,key=lambda x:x[-1])
#rres = [x[:-1]+[str(round(x[-1],4))] for x in res]
#for l in rres[:20]:
#    print(','.join(l))





