# -*- coding: utf-8 -*-

import dgl
import torch
import pickle
from time import time
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

class HeteroDotPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']

class MLPPredictor(nn.Module):
    def __init__(self, h_feats, mtype):
        super().__init__()
        self.mtype = mtype
        if mtype == 'rrm':
            self.W1 = nn.Linear(h_feats * 2, h_feats)
            self.W2 = nn.Linear(h_feats, 1)
        elif mtype == 'rm':
            self.W1 = nn.Linear(h_feats * 2, h_feats)
            self.W2 = nn.Linear(h_feats, h_feats)
            self.W3 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        if self.mtype == 'rrm':
            return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}
        elif self.mtype == 'rm':
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

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        graph.ndata['h'] = h
        return h

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, mtype):
        super().__init__()
        self.mtype = mtype
        if mtype == 'rm':
            self.pred = MLPPredictor(in_features, mtype)
        elif mtype == 'rrm':
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = MLPPredictor(out_features, mtype)
        elif mtype == 'rrd':
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotPredictor()
    def forward(self, g, neg_g, x, etype):
        if self.mtype == 'rm':
            return self.pred(g, x, etype), self.pred(neg_g, x, etype)
        else:
            h = self.sage(g, x)
            return self.pred(g, h, etype), self.pred(neg_g, h, etype)

def compute_loss(pos_score, neg_score):
    scores = torch.reshape(torch.cat([pos_score, neg_score]), (-1,))
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_acc(pos_score, neg_score):
    return torch.sigmoid(pos_score).mean().item(), 1- torch.sigmoid(neg_score).mean().item()

def main(mtype, etype):
    n_features = 64
    with open('splieted_embgraph.pkl', 'br') as file:
        train_g, test_g = pickle.load(file)
    train_g = train_g.to('cuda:0')
    test_g = test_g.to('cuda:0')
        
    node_types = train_g.ntypes
    for k in node_types:
        train_g.nodes[k].data['feature'] = test_g.nodes[k].data['feature'] = torch.randn(train_g.num_nodes(k), n_features).to('cuda:0') 

    model = Model(n_features, 64, 10, train_g.etypes, mtype).to('cuda:0')
    train_node_features = {k:train_g.nodes[k].data['feature'] for k in node_types}
    test_node_features = {k:test_g.nodes[k].data['feature'] for k in node_types}
    opt = torch.optim.Adam(model.parameters())

    edge = {'dg':('Drug', 'change', 'Gene'),
            'gg':('Gene', 'SLpair', 'Gene'),
            'gc':('Cancer', 'due_to', 'Gene')}[etype]
    
    max_test_pos_score = 0
    start = time()
    
    for epoch in range(501):
        train_negative_graph = construct_negative_graph(train_g, 10, edge)
        train_pos_score, train_neg_score = model(train_g, train_negative_graph, train_node_features, edge)
        loss = compute_loss(train_pos_score.cpu(), train_neg_score.cpu())
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            test_negative_graph = construct_negative_graph(test_g, 10, edge)
            test_pos_score, test_neg_score = model(test_g, test_negative_graph, test_node_features, edge)
            test_pos_acc, _ = compute_acc(test_pos_score.cpu(), test_neg_score.cpu())
            if test_pos_acc > max_test_pos_score:
                max_test_pos_score = test_pos_acc
                max_score_epoch = epoch
                time_save = time()-start
                data_save = test_pos_score, test_neg_score
                
    with open("./scoreData/{}_{}_{}.pkl".format(mtype, etype, int(time())), "bw") as file:
        pickle.dump(((max_score_epoch, time_save),data_save), file)
    print(max_score_epoch, max_test_pos_score)
    
if __name__ == "__main__":
    for m in ['rrm']:
        for e in ['dg','gg','gc']:
            for i in range(11):
                main(m ,e)