# -*- coding: utf-8 -*-

import xlrd
import pickle

def loadData(fileName, headName, tailName, weightName):
    table = xlrd.open_workbook(fileName).sheets()[0]
    title = table.row_values(0)
    h, t, w = title.index(headName), title.index(tailName), title.index(weightName)
    return table.col_values(h, 1), table.col_values(t, 1), table.col_values(w, 1)

geneA, geneB, ggWeight = loadData('SL_A_B_Score.xls', 'GeneASymbol', 'GeneBSymbol', 'SL_score')
geneC, cancer, gcWeight = loadData('SL_mutGene_TSG_ReCancer.xls', 'MutationGene', 'Repositioning_of_the_Cancer', 'ReCancerScore')
drug, geneD, dgWeight = loadData('SL_TargetGene_Drug_KnowDisease_Score.xls', 'DrugName', 'TargetGene', 'DrugScore')

geneSet = set(geneA) | set(geneB)
drugSet = set(drug)
cancerSet = set(cancer)
itemSet = geneSet | drugSet | cancerSet
itemIdDict = {k:i for i,k in enumerate(itemSet)}

with open('graph.txt', 'w') as file:
    file.write(''.join('{} {} {}\n'.format(itemIdDict[l], itemIdDict[r], w) 
                        for l, r, w in zip(geneA, geneB, ggWeight)))
    file.write(''.join('{} {} {}\n'.format(itemIdDict[l], itemIdDict[r], w) 
                        for l, r, w in zip(cancer, geneC, gcWeight)))
    file.write(''.join('{} {} {}\n'.format(itemIdDict[l], itemIdDict[r], w) 
                        for l, r, w in zip(geneD, drug, dgWeight)))
with open('txtGraphIdDict.pkl', 'bw') as file:
    pickle.dump({'g':{x:itemIdDict[x] for x in geneSet}, 
                 'd':{x:itemIdDict[x] for x in drugSet}, 
                 'c':{x:itemIdDict[x] for x in cancerSet}}, file)