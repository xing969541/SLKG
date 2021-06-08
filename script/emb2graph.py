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

training_rate = 0.9

trainGraphDataDict = {}
trainGraphWeightDict = {}
testGraphDataDict = {}
testGraphWeightDict = {}
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

trainMask = torch.rand(len(geneA)) < training_rate
testMask = trainMask == False
nodeNum['Gene'] = len(geneSet)
trainGraphDataDict[('Gene', 'SLpair', 'Gene')] = (
    torch.cat((geneA[trainMask], geneB[trainMask])), 
    torch.cat((geneB[trainMask], geneA[trainMask]))
    )
trainGraphWeightDict[('Gene', 'SLpair', 'Gene')] = torch.cat((ggWeight[trainMask], ggWeight[trainMask]))
testGraphDataDict[('Gene', 'SLpair', 'Gene')] = (
    torch.cat((geneA[testMask], geneB[testMask])), 
    torch.cat((geneB[testMask], geneA[testMask]))
    )
testGraphWeightDict[('Gene', 'SLpair', 'Gene')] = torch.cat((ggWeight[testMask], ggWeight[testMask]))

# Process gene-cancer form.
gene, cancer, gcWeight = loadData('SL_mutGene_TSG_ReCancer.xls', 'MutationGene', 'Repositioning_of_the_Cancer', 'ReCancerScore')
gcWeight = torch.tensor(gcWeight)
cancerSet = set(cancer)
cancerFeature = torch.tensor([emb[nodeId2embIndexDict[name2nodeIdDict['c'][c]]] for c in cancerSet])
cancerIdDict = {c:i for i, c in enumerate(cancerSet)}
gene = torch.from_numpy(np.array([geneIdDict[g] for g in gene]))
cancer = torch.from_numpy(np.array([cancerIdDict[c] for c in cancer]))

trainMask = torch.rand(len(cancer)) < training_rate
testMask = trainMask == False
nodeNum['Cancer'] = len(cancerSet)
trainGraphDataDict[('Gene', 'lead_to', 'Cancer')] = (gene[trainMask], cancer[trainMask])
trainGraphDataDict[('Cancer', 'due_to', 'Gene')] = (cancer[trainMask], gene[trainMask])
trainGraphWeightDict[('Gene', 'lead_to', 'Cancer')] = gcWeight[trainMask]
trainGraphWeightDict[('Cancer', 'due_to', 'Gene')] = gcWeight[trainMask]
testGraphDataDict[('Gene', 'lead_to', 'Cancer')] = (gene[testMask], cancer[testMask])
testGraphDataDict[('Cancer', 'due_to', 'Gene')] = (cancer[testMask], gene[testMask])
testGraphWeightDict[('Gene', 'lead_to', 'Cancer')] = gcWeight[testMask]
testGraphWeightDict[('Cancer', 'due_to', 'Gene')] = gcWeight[testMask]

# Process gene-drug form.
drug, gene, dgWeight = loadData('SL_TargetGene_Drug_KnowDisease_Score.xls', 'DrugName', 'TargetGene', 'DrugScore')
dgWeight = torch.tensor(dgWeight)
drugSet = set(drug)
drugIdDict = {d:i for i, d in enumerate(drugSet)}
gene = torch.from_numpy(np.array([geneIdDict[g] for g in gene]))
drug = torch.from_numpy(np.array([drugIdDict[d] for d in drug]))
drugFeature = torch.tensor([emb[nodeId2embIndexDict[name2nodeIdDict['d'][d]]] for d in drugSet])

trainMask = torch.rand(len(drug)) < training_rate
testMask = trainMask == False
nodeNum['Drug'] = len(drugSet)
trainGraphDataDict[('Gene', 'changed_by', 'Drug')] = (gene[trainMask], drug[trainMask])
trainGraphDataDict[('Drug', 'change', 'Gene')] = (drug[trainMask], gene[trainMask])
trainGraphWeightDict[('Gene', 'changed_by', 'Drug')] = dgWeight[trainMask]
trainGraphWeightDict[('Drug', 'change', 'Gene')] = dgWeight[trainMask]
testGraphDataDict[('Gene', 'changed_by', 'Drug')] = (gene[testMask], drug[testMask])
testGraphDataDict[('Drug', 'change', 'Gene')] = (drug[testMask], gene[testMask])
testGraphWeightDict[('Gene', 'changed_by', 'Drug')] = dgWeight[testMask]
testGraphWeightDict[('Drug', 'change', 'Gene')] = dgWeight[testMask]

# Create hetero graph, save necessary data.
trainG = dgl.heterograph(trainGraphDataDict, idtype=torch.int64, num_nodes_dict=nodeNum)
trainG.edata['weight'] = trainGraphWeightDict
testG = dgl.heterograph(testGraphDataDict, idtype=torch.int64, num_nodes_dict=nodeNum)
testG.edata['weight'] = testGraphWeightDict

trainG.nodes['Gene'].data['feature'] = testG.nodes['Gene'].data['feature'] = geneFeature
trainG.nodes['Drug'].data['feature'] = testG.nodes['Drug'].data['feature'] = drugFeature
trainG.nodes['Cancer'].data['feature'] = testG.nodes['Cancer'].data['feature'] = cancerFeature
print('Train Graph:')
print(trainG)
print('Test Graph:')
print(testG)

with open('splieted_embgraph.pkl', 'bw') as file:
    pickle.dump((trainG, testG), file)
with open('splieted_embidDict.pkl', 'bw') as file:
    pickle.dump({'gene':geneIdDict, 'drug':drugIdDict, 'cancer':cancerIdDict}, 
                file)