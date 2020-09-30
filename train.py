import os
import sys
from dataloader import PseudoTokenMappingDataset, custom_collate_fn
from transformers import BertModel, BertTokenizer, AdamW
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np
import random

import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.xla_multiprocessing as xmp 
import torch_xla.distributed.parallel_loader as pl


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
       # Sets a common random seed - both for initialization and ensuring graph is the same
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed) # this should be done before firing Python itself
       np.random.seed(seed)
       torch.manual_seed(seed)

def train_loop(dl_1, dl_2, model, model_orig, optimizer, tokenizer, device, epoch, loss_fn):
    model.train()
    xm.master_print("Epoch ",epoch,"/",10)
    for dataloader in [dl_1,dl_2]:
        for i,(sent1,sent2,tok_maps) in enumerate(dataloader):
            enc_1 = tokenizer(sent1,padding=True,truncation=True,return_tensors='pt',max_length=512).to(device)
            enc_2 = tokenizer(sent2,padding=True,truncation=True,return_tensors='pt',max_length=512).to(device)
            embeddings_1 = model(**enc_1).last_hidden_state
            embeddings_2 = model(**enc_2).last_hidden_state
            embeddings_orig_1 = model_orig(**enc_1).last_hidden_state
            embeddings_orig_2 = model_orig(**enc_2).last_hidden_state
            optimizer.zero_grad()
            loss = 0
            for j,tok_map in enumerate(tok_maps):
                emb_1 = embeddings_1[j]
                emb_orig_1 = embeddings_orig_1[j]
                emb_2 = embeddings_2[j]
                emb_orig_2 = embeddings_orig_2[j]
                max_length_1 = emb_1.shape[0]
                max_length_2 = emb_2.shape[0] 
                tokens_1 = [x[0] for x in tok_map if x[0] < max_length_1]
                tokens_2 = [x[1] for x in tok_map if x[1] < max_length_2]
                min_toks = min(len(tokens_1),len(tokens_2))
                loss += loss_fn(emb_1[tokens_1[:min_toks],:],emb_2[tokens_2[:min_toks],:])
                loss += loss_fn(emb_1,emb_orig_1)
                loss += loss_fn(emb_2,emb_orig_2)
            loss.backward()
            xm.optimizer_step(optimizer,barrier=True)
            
            if (i+1)%100 == 0:
                    print('loss:{}'.format(loss.item()),flush=True)
                    xm.save(model.state_dict(),'ckpt/bert_model.hdf5')
    




def train(l1,l2,batch_size):
    seed_everything(1234)
    device = xm.xla_device()
    token_mapping_datset_1 = PseudoTokenMappingDataset(l1+'_sentences.txt',l2+'_pseudo_'+l1+'.txt')
    token_mapping_datset_2 = PseudoTokenMappingDataset(l2+'_sentences.txt',l1+'_pseudo_'+l2+'.txt')
    # token_mapping_sampler_1 = torch.utils.data.distributed.DistributedSampler(token_mapping_datset_1,num_replicas=xm.xrt_world_size(),rank= xm.get_ordinal(), shuffle=False)
    # token_mapping_sampler_2 = torch.utils.data.distributed.DistributedSampler(token_mapping_datset_2,num_replicas=xm.xrt_world_size(),rank= xm.get_ordinal(), shuffle=False)
    token_mapping_dl_1 = DataLoader(token_mapping_datset_1,batch_size=batch_size,collate_fn=custom_collate_fn)
    token_mapping_dl_2 = DataLoader(token_mapping_datset_2,batch_size=batch_size,collate_fn=custom_collate_fn) 
    model = BertModel.from_pretrained('bert-base-multilingual-cased',return_dict=True)
    model_orig = BertModel.from_pretrained('bert-base-multilingual-cased',return_dict=True)
    model = model.to(device)
    model_orig = model_orig.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    optim = AdamW(model.parameters(),lr=5e-5)
    mse_loss = MSELoss()
    model.train()
    iterations=0
    for epoch in range(10):
        # dl_1 = pl.ParallelLoader(token_mapping_dl_1,[device]).per_device_loader(device)
        # dl_2 = pl.ParallelLoader(token_mapping_dl_2,[device]).per_device_loader(device)
        train_loop(token_mapping_dl_1,token_mapping_dl_2,model,model_orig,optim,tokenizer,device,epoch,mse_loss)
        



if __name__ == "__main__":
    seed_everything
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    batch_size = int(sys.argv[3])
    # xmp.spawn(train,args=(l1,l2,batch_size,),nprocs=8,start_method='fork')
    train(l1,l2,batch_size)


        



        



