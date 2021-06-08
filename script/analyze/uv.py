# -*- coding: utf-8 -*-

import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, dim1, dim2, d):
        super(Model,self).__init__()
        u = torch.randn((dim1, d), requires_grad=True, dtype=torch.float64)
        v = torch.randn((d, dim2), requires_grad=True, dtype=torch.float64)
        self.u = torch.nn.Parameter(u)
        self.v = torch.nn.Parameter(v)
    def forward(self):
        return torch.sigmoid(torch.mm(self.u, self.v))

def computeLoss(score, target):
    return F.mse_loss(score, target, reduction='mean')

def main(etype):
    with open("PredData_{}.pkl".format(etype), "br") as file:
        (src, dst), mat = pickle.load(file)
    src, dst = src.detach().cpu().numpy(), dst.detach().cpu().numpy()
    mat = torch.tensor(mat).to('cuda:0')
    target = np.zeros(mat.shape)
    target[src,dst] = 1
    target = torch.tensor(target).to('cuda:0')
    
    model = Model(*mat.shape, 32).to('cuda:0')
    opt = torch.optim.Adam(model.parameters(), 
                           lr=1e-1, weight_decay=1e-5)

    lossSave = 1e10
    for epoch in range(76):
        score = model()
        loss = computeLoss(score, target)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss < lossSave:
            lossSave = loss.item()
            scoreSave = score.detach().cpu().numpy()
        if not epoch % 25:
            print("epoch:{}\tloss:{}".format(epoch, loss.item()))
    np.save("uvdata_{}".format(etype), scoreSave)
    
if __name__ == "__main__":
    main('gg')
#    main('dg')
#    main('gc')