import torch
from torch.utils.data import IterableDataset,DataLoader, Dataset
from utils import get_token_mappings_line
import os
import sys

class PseudoTokenMappingDataset(IterableDataset):
    def __init__(self,file1,file2):
        self.file1 = file1
        self.file2 = file2

    def strip_line(self,line):
        line_strip = line.replace("\n","")
        line_strip = line_strip.replace("_"," ")
        return line_strip

    def preprocess(self,line1,line2):
        line1_strip = self.strip_line(line1)
        line2_strip = self.strip_line(line2)
        tok_maps = get_token_mappings_line(line1,line2)
        return line1_strip, line2_strip, tok_maps

    def __iter__(self):
        l1_iter = open(self.file1)
        l2_iter = open(self.file2)
        tok_maps = map(self.preprocess,l1_iter,l2_iter)
        return tok_maps

        
def custom_collate_fn(batch):
    sentences_1 = [x[0] for x in batch]
    sentences_2 = [x[1] for x in batch]
    token_mappings = [x[2] for x in batch]
    return sentences_1,sentences_2,token_mappings

    
if __name__=="__main__":
    dataset = PseudoTokenMappingDataset('original_hindi.txt','pseudo_marathi.txt')
    dl = DataLoader(dataset=dataset,batch_size=5,collate_fn=custom_collate_fn)
    for i,x in enumerate(dl):
        print(x[0])
        if i==0:
            break
    