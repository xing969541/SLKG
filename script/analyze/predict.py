# -*- coding: utf-8 -*-

import os
import dgl
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, h_feats)
        self.W3 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W3(F.relu(self.W2(F.relu(self.W1(h))))).squeeze(1)}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.pred = MLPPredictor(in_features)
    def forward(self, g, neg_g, x, etype):
        return self.pred(g, x, etype), self.pred(neg_g, x, etype)
    def predict(self, g, x, etype):
        return self.pred(g, x, etype)
    
def construct_all_connect_graph(graph, etype):
    utype, _, vtype = etype
    graph = graph.cpu()
    src, dst = graph.edges(etype=etype)
    if utype == vtype:
        set_ = set(x.item() for x in src) & set(x.item() for x in dst)
        node = torch.tensor(list(set_))
        node = node.repeat(node.shape[0], 1)
        src = node.reshape(-1)
        dst = node.t().reshape(-1)
    else:
        src = torch.tensor(list(set(x.item() for x in src)))
        dst = torch.tensor(list(set(x.item() for x in dst)))
        src, dst = src.repeat(dst.shape[0], 1), dst.repeat(src.shape[0], 1)
        src, dst = src.reshape(-1), dst.t().reshape(-1)
    length = len(src)
    step = (length // 100) + 1
    current = 0
    i=0
    print("0/100", end='')
    while current < length:
        tmp = current+step
        yield dgl.heterograph({etype: (src[current:tmp], dst[current:tmp])},
            num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}).to('cuda:0')
        current = tmp
        i+=1
        print("\r{}/100".format(i), end='')
    print()

class Data:
    def __init__(self, x, y):
        self._mat = np.zeros((x,y))
    
    def update(self, graph, score, etype):
        src, dst = graph.edges(etype=etype)
        src = src.detach().cpu().numpy()
        dst = dst.detach().cpu().numpy()
        self._mat[src,dst] = torch.sigmoid(score).detach().cpu().numpy()
    
    @property
    def mat(self):
        if self._mat.shape[0] == self._mat.shape[1]:
            tmp = np.arange(self._mat.shape[0])
            self._mat[tmp, tmp] = 0
        return self._mat

def main(etype):
    edge = {'dg':('Drug', 'change', 'Gene'),
            'gg':('Gene', 'SLpair', 'Gene'),
            'gc':('Cancer', 'due_to', 'Gene')}[etype]
    
    for fn in os.listdir():
        head = "model_{}".format(etype)
        if fn.startswith(head):
            model = torch.load(fn).to("cuda:0")
            break
    
    with open("embgraph.pkl", "br") as file:
        G = pickle.load(file).to("cuda:0")
        
    num_nodes_dict={ntype: G.num_nodes(ntype) for ntype in G.ntypes}
    node_features = {k:G.nodes[k].data['feature'].to("cuda:0") for k in G.ntypes}
    data = Data(num_nodes_dict[edge[0]], num_nodes_dict[edge[-1]])
    for g in construct_all_connect_graph(G, edge):
        score = model.predict(g, node_features, edge)
        data.update(g, score, edge)
    with open("PredData_{}.pkl".format(etype), "bw") as file:
        pickle.dump((G.edges(etype=edge), data.mat), file)

if __name__ == "__main__":
    main('dg')
    main('gg')
    main('gc')