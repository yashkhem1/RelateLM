import torch
from torch.utils.data import IterableDataset,DataLoader, Dataset, ConcatDataset
import os
import sys
import json
import pickle
import numpy as np

class WikipediaTokenMapDataset(Dataset):
    def __init__(self,json_files):
        self.dict_list = []
        if isinstance(json_files,list):
            for file_ in json_files:
                with open(file_,'r') as f:
                    try:
                        json_ = json.load(f)
                    except json.JSONDecodeError as err:
                        # grab a reasonable section, say 40 characters.
                        start, stop = max(0, err.pos - 40), err.pos + 40
                        snippet = err.doc[start:stop]
                        print(err)
                        print('... ' if start else '', snippet, ' ...' if stop < len(err.doc) else '', sep="")
                        print('^'.rjust(21 if not start else 25))
                        raise(ValueError)
                    self.dict_list += json_['data']
                print(file_,"loaded")
        else:
            with open(json_files,'r') as f:
                try:
                    json_ = json.load(f)
                except json.JSONDecodeError as err:
                    # grab a reasonable section, say 40 characters.
                    start, stop = max(0, err.pos - 20), err.pos + 20
                    snippet = err.doc[start:stop]
                    print(err)
                    print('... ' if start else '', snippet, ' ...' if stop < len(err.doc) else '', sep="")
                    print('^'.rjust(21 if not start else 25))
                    raise(ValueError)
                self.dict_list += json_['data']

        del json_

    def __len__(self):
        return len(self.dict_list)

    def __getitem__(self,index):
        return self.dict_list[index]['token_ids_1'],self.dict_list[index]['token_ids_2'],self.dict_list[index]['token_maps_1'],self.dict_list[index]['token_maps_2']
        # return torch.Tensor(self.dict_list[index]['token_ids_1']),torch.Tensor(self.dict_list[index]['token_ids_2'])

class BilingualWithNegativeSampling(Dataset):
    def __init__(self,dict_files,neg_samples):
        self.neg_samples = neg_samples
        dict_list =[]
        if isinstance(dict_files,list):
            for file_ in dict_files:
                with open(file_,'r') as f:
                    try:
                        dict_list += json.load(f)['data']     
                    except json.JSONDecodeError as err:
                        # grab a reasonable section, say 40 characters.
                        start, stop = max(0, err.pos - 20), err.pos + 20
                        snippet = err.doc[start:stop]
                        print(err)
                        print('... ' if start else '', snippet, ' ...' if stop < len(err.doc) else '', sep="")
                        print('^'.rjust(21 if not start else 25))
                        raise(ValueError)
        else:
            with open(dict_files,'r') as f:
                try:
                    dict_list = json.load(f)['data']
                except json.JSONDecodeError as err:
                    # grab a reasonable section, say 40 characters.
                    start, stop = max(0, err.pos - 20), err.pos + 20
                    snippet = err.doc[start:stop]
                    print(err)
                    print('... ' if start else '', snippet, ' ...' if stop < len(err.doc) else '', sep="")
                    print('^'.rjust(21 if not start else 25))
                    raise(ValueError)

        self.l1_tok_ids = [x['token_ids_1'] for x in dict_list]
        self.l2_tok_ids = [x['token_ids_2'] for x in dict_list]
        self.len = len(self.l1_tok_ids)
        del dict_list


    def __len__(self):
        return self.len

    def __getitem__(self,index):
        item_list = [torch.tensor(self.l1_tok_ids[index],dtype=torch.long),torch.tensor(self.l2_tok_ids[index],dtype=torch.long)]
        # negative_indices = np.random.choice(np.arange(self.len),self.neg_samples,replace=False)
        negative_indices = np.random.randint(0,self.len,self.neg_samples)
        for neg_index in negative_indices:
            item_list.append(torch.tensor(self.l1_tok_ids[neg_index],dtype=torch.long))
            item_list.append(torch.tensor(self.l2_tok_ids[neg_index],dtype=torch.long))
        return item_list
        
def custom_collate_fn(batch):
    sentences_1 = [x[0] for x in batch]
    sentences_2 = [x[1] for x in batch]
    token_mappings = [x[2] for x in batch]
    return sentences_1,sentences_2,token_mappings

def token_maps_collate(batch):
    tok_ids_1 = torch.tensor([x[0] for x in batch],dtype=torch.long)
    tok_ids_2 = torch.tensor([x[1] for x in batch],dtype=torch.long)
    tok_maps_1 = torch.tensor([x[2] for x in batch])
    tok_maps_2 = torch.tensor([x[3] for x in batch])
    att_masks_1 = (tok_ids_1>0).type(torch.long)
    att_masks_2 = (tok_ids_2>0).type(torch.long)
    batch_size,max_len = tok_ids_1.size()
    weights = (tok_maps_1>0).view(-1,1).type(torch.float)
    flat_maps_1 = (tok_maps_1 + max_len*(torch.arange(batch_size).view(-1,1))).view(-1).type(torch.long)
    flat_maps_2 = (tok_maps_2 + max_len*(torch.arange(batch_size).view(-1,1))).view(-1).type(torch.long)
    del tok_maps_1, tok_maps_2
    return tok_ids_1, tok_ids_2, flat_maps_1, flat_maps_2, att_masks_1, att_masks_2, weights
    
if __name__=="__main__":
    # dataset = BilingualWithNegativeSampling(['preprocessed_data/Hindi_Punjabi_trans_Hindi_bilingual.json'],2)
    dataset = WikipediaTokenMapDataset(['preprocessed_data/Hindi_Punjabi_trans_Hindi_wik.json'])
    dl = DataLoader(dataset=dataset,batch_size=2,shuffle=False,collate_fn=token_maps_collate)
    for i,b in enumerate(dl):
        print(b)
        break
    