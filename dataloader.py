import torch
from torch.utils.data import IterableDataset,DataLoader, Dataset, ConcatDataset
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
        l1_iter = open(self.file1,encoding='utf-8')
        l2_iter = open(self.file2,encoding='utf-8')
        tok_maps = map(self.preprocess,l1_iter,l2_iter)
        return tok_maps


        
def custom_collate_fn(batch):
    sentences_1 = [x[0] for x in batch]
    sentences_2 = [x[1] for x in batch]
    token_mappings = [x[2] for x in batch]
    return sentences_1,sentences_2,token_mappings

    
if __name__=="__main__":
    dataset = PseudoTokenMappingDataset('Hindi_sentences.txt','Marathi_pseudo_Hindi.txt')
    dl = DataLoader(dataset=dataset,batch_size=5,collate_fn=custom_collate_fn)
    for i,(x,y,z) in enumerate(dl):
        print(x)
        if i==0:
            break
    