# -*- coding: utf-8 -*-

import dgl
import xlrd
import torch
import pickle
import numpy as np

emb = np.load('emb.npy')
with open('nodeid_to_index.pickle', 'br') as file:
    nodeId2embIndexDict = pickle.load(file)
with open('txtGraphIdDict.pkl', 'br') as file:
    name2nodeIdDict = pickle.load(file)

graphDataDict = {}
graphWeightDict = {}
nodeNum = {}

def loadData(fileName, headName, tailName, weightName):
    table = xlrd.open_workbook(fileName).sheets()[0]
    title = table.row_values(0)
    h, t, w = title.index(headName), title.index(tailName), title.index(weightName)
    return table.col_values(h, 1), table.col_values(t, 1), [float(x) for x in table.col_values(w, 1)]

# Process gene-gene form.
geneA, geneB, ggWeight = loadData('SL_A_B_Score.xls', 'GeneASymbol', 'GeneBSymbol', 'SL_score')
ggWeight = torch.tensor(ggWeight)
geneSet = set(geneA) | set(geneB)
geneIdDict = {g:i for i, g in enumerate(geneSet)}
geneFeature = torch.tensor([emb[nodeId2embIndexDict[name2nodeIdDict['g'][g]]] for g in geneSet])
geneA = torch.from_numpy(np.array([geneIdDict[g] for g in geneA]))
geneB = torch.from_numpy(np.array([geneIdDict[g] for g in geneB]))

nodeNum['Gene'] = len(geneSet)
graphDataDict[('Gene', 'SLpair', 'Gene')] = (
    torch.cat((geneA, geneB)), 
    torch.cat((geneB, geneA))
    )
graphWeightDict[('Gene', 'SLpair', 'Gene')] = torch.cat((ggWeight, ggWeight))

# Process gene-cancer form.
gene, cancer, gcWeight = loadData('SL_mutGene_TSG_ReCancer.xls', 'MutationGene', 'Repositioning_of_the_Cancer', 'ReCancerScore')
gcWeight = torch.tensor(gcWeight)
cancerSet = set(cancer)
cancerFeature = torch.tensor([emb[nodeId2embIndexDict[name2nodeIdDict['c'][c]]] for c in cancerSet])
cancerIdDict = {c:i for i, c in enumerate(cancerSet)}
gene = torch.from_numpy(np.array([geneIdDict[g] for g in gene]))
cancer = torch.from_numpy(np.array([cancerIdDict[c] for c in cancer]))

nodeNum['Cancer'] = len(cancerSet)
graphDataDict[('Gene', 'lead_to', 'Cancer')] = (gene, cancer)
graphDataDict[('Cancer', 'due_to', 'Gene')] = (cancer, gene)
graphWeightDict[('Gene', 'lead_to', 'Cancer')] = gcWeight
graphWeightDict[('Cancer', 'due_to', 'Gene')] = gcWeight

# Process gene-drug form.
drug, gene, dgWeight = loadData('SL_TargetGene_Drug_KnowDisease_Score.xls', 'DrugName', 'TargetGene', 'DrugScore')
dgWeight = torch.tensor(dgWeight)
drugSet = set(drug)
drugIdDict = {d:i for i, d in enumerate(drugSet)}
gene = torch.from_numpy(np.array([geneIdDict[g] for g in gene]))
drug = torch.from_numpy(np.array([drugIdDict[d] for d in drug]))
drugFeature = torch.tensor([emb[nodeId2embIndexDict[name2nodeIdDict['d'][d]]] for d in drugSet])

nodeNum['Drug'] = len(drugSet)
graphDataDict[('Gene', 'changed_by', 'Drug')] = (gene, drug)
graphDataDict[('Drug', 'change', 'Gene')] = (drug, gene)
graphWeightDict[('Gene', 'changed_by', 'Drug')] = dgWeight
graphWeightDict[('Drug', 'change', 'Gene')] = dgWeight

# Create hetero graph, save necessary data.
g = dgl.heterograph(graphDataDict, idtype=torch.int64, num_nodes_dict=nodeNum)
g.edata['weight'] = graphWeightDict

g.nodes['Gene'].data['feature'] = geneFeature
g.nodes['Drug'].data['feature'] = drugFeature
g.nodes['Cancer'].data['feature'] = cancerFeature
print('Graph:')
print(g)

with open('embgraph.pkl', 'bw') as file:
    pickle.dump(g, file)
with open('embidDict.pkl', 'bw') as file:
    pickle.dump({'gene':geneIdDict, 'drug':drugIdDict, 'cancer':cancerIdDict}, 
                file)