# -*- coding: utf-8 -*-

import dgl
import torch
import pickle
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

def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    graph = graph.cpu()
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}).to('cuda:0')

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.pred = MLPPredictor(in_features)
    def forward(self, g, neg_g, x, etype):
        return self.pred(g, x, etype), self.pred(neg_g, x, etype)
    def predict(self, g, x, etype):
        return self.pred(g, x, etype)

def compute_loss(pos_score, neg_score):
    scores = torch.reshape(torch.cat([pos_score, neg_score]), (-1,))
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def computeF1(pos_score, neg_score):
    TP = (torch.sigmoid(pos_score)>0.5).sum().item()+1e-8
    FN = (torch.sigmoid(pos_score)<0.5).sum().item()+1e-8
    FP = (torch.sigmoid(neg_score)>0.5).sum().item()+1e-8
    
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    return 2 * r * p / (r + p)

def main(etype):
    with open('embgraph.pkl', 'br') as file:
        g = pickle.load(file)
    g = g.to('cuda:0')
    
    node_types = g.ntypes
    n_features = g.nodes['Gene'].data['feature'].shape[1]
    
    model = Model(n_features, 64, 10, g.etypes).to('cuda:0')
    node_features = {k:g.nodes[k].data['feature'] for k in node_types}
    opt = torch.optim.Adam(model.parameters())

    edge = {'dg':('Drug', 'change', 'Gene'),
            'gg':('Gene', 'SLpair', 'Gene'),
            'gc':('Cancer', 'due_to', 'Gene')}[etype]
    
    max_f1 = 0.97
    
    for epoch in range(1001):
        negative_graph = construct_negative_graph(g, 1, edge)
        pos_score, neg_score = model(g, negative_graph, node_features, edge)
        loss = compute_loss(pos_score.cpu(), neg_score.cpu())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        F1 = computeF1(pos_score, neg_score)
        if F1 > max_f1:
            torch.save(model, "./model_{}_{}.pkl".format(etype, F1))
            dataSave = (pos_score, neg_score)
            max_f1 = F1
            
    with open("./data_{}_{}.pkl".format(etype, F1), "bw") as file:
        pickle.dump(dataSave, file)
        
if __name__ == "__main__":
    for e in ['dg','gg','gc']:
        main(e)