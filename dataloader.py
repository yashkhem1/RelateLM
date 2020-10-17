import torch
from torch.utils.data import IterableDataset,DataLoader, Dataset, ConcatDataset
import os
import sys
import json

class WikipediaTokenMapDataset(Dataset):
    def __init__(self,json_files):
        self.dict_list = []
        if isinstance(json_files,list):
            for file_ in json_files:
                with open(file_,'r') as f:
                    json_ = json.load(f)
                    self.dict_list += json_['data']
        else:
            with open(json_files,'r') as f:
                json_ = json.load(f)
                self.dict_list += json_['data']

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self,index):
        dict_i = self.dict_list[index]
        return dict_i['token_ids_1'],dict_i['token_ids_2'],dict_i['token_maps_1'],dict_i['token_maps_2']


        
def custom_collate_fn(batch):
    sentences_1 = [x[0] for x in batch]
    sentences_2 = [x[1] for x in batch]
    token_mappings = [x[2] for x in batch]
    return sentences_1,sentences_2,token_mappings

def token_maps_collate(batch):
    tok_ids_1 = torch.tensor([x[0] for x in batch])
    tok_ids_2 = torch.tensor([x[1] for x in batch])
    tok_maps_1 = torch.tensor([x[2] for x in batch])
    tok_maps_2 = torch.tensor([x[3] for x in batch])
    att_masks_1 = tok_ids_1>0
    att_masks_2 = tok_ids_2>0
    batch_size,max_len = tok_ids_1.size()
    weights_1 = (tok_maps_1>0).type(torch.int).view(-1,1)
    weights_2 = (tok_maps_2>0).type(torch.int).view(-1,1)
    flat_maps_1 = (tok_maps_1 + max_len*(torch.arange(batch_size).view(-1,1))).view(-1)
    flat_maps_2 = (tok_maps_2 + max_len*(torch.arange(batch_size).view(-1,1))).view(-1)
    return tok_ids_1, tok_ids_2, flat_maps_1, flat_maps_2, att_masks_1, att_masks_2, weights_1, weights_2
    
if __name__=="__main__":
    dataset = WikipediaTokenMapDataset('Marathi_Hindi.json')
    dl = DataLoader(dataset=dataset,batch_size=5,shuffle=False, collate_fn= token_maps_collate)
    for i,(x,y,z,w,a,b,c,d) in enumerate(dl):
        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(w.shape)
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(z.shape)
        break
    