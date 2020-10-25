import os
import sys
from dataloader import WikipediaTokenMapDataset, token_maps_collate
from transformers import BertModel, AdamW, BertConfig
from utils import load_BFTC_from_TF_ckpt
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import numpy as np
import random
import time

import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.xla_multiprocessing as xmp 
import torch_xla.distributed.parallel_loader as pl
import gc
import argparse


os.environ['XLA_USE_BF16']="1"

def seed_everything(seed):
       # Sets a common random seed - both for initialization and ensuring graph is the same
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed) # this should be done before firing Python itself
       np.random.seed(seed)
       torch.manual_seed(seed)

def train_loop(dl, model, optimizer, device, epoch,ckpt,print_every):
    tracker = xm.RateTracker()
    model.train()
    xm.master_print("Epoch ",epoch,"/",10)
    loss_fn = MSELoss()
    for i,(tok_ids_1,tok_ids_2,flat_maps_1,flat_maps_2, att_masks_1, att_masks_2,weights_1, weights_2) in enumerate(dl):
        optimizer.zero_grad()
        # tok_ids_1,tok_ids_2, att_masks_1, att_masks_2= tok_ids_1.to(device,dtype=torch.long), tok_ids_2.to(device,dtype=torch.long), att_masks_1.to(device,dtype=torch.long), att_masks_2.to(device,dtype=torch.long)
        # flat_maps_1,flat_maps_2,weights_1,weights_2 = flat_maps_1.to(device,dtype=torch.long),flat_maps_2.to(device,dtype=torch.long),weights_1.to(device,dtype=torch.float),weights_2.to(device,dtype=torch.float)
        embeddings_1 = model(input_ids = tok_ids_1, attention_mask= att_masks_1)[0]
        embeddings_2 = model(input_ids = tok_ids_2, attention_mask= att_masks_2)[0]
        embeddings_orig_1 = model_orig(input_ids = tok_ids_1, attention_mask= att_masks_1)[0]
        embeddings_orig_2 = model_orig(input_ids = tok_ids_2, attention_mask= att_masks_2)[0]
        batch_size,max_length,hidden_size = embeddings_1.shape
        tracker.add(batch_size)
        flat_embeddings_1 = embeddings_1.view(-1,hidden_size)
        flat_embeddings_2 = embeddings_2.view(-1,hidden_size)
        loss = loss_fn(flat_embeddings_1[flat_maps_1]*weights_1,flat_embeddings_2[flat_maps_2]*weights_2)
        loss += loss_fn(embeddings_1,embeddings_orig_1)
        loss += loss_fn(embeddings_2,embeddings_orig_2)
        loss.backward()
        xm.optimizer_step(optimizer)
        
        if (i+1)%print_every == 0:
            xm.master_print('[xla:{}], Loss: {}, Iteration: {}, Time:{}s, Rate={:.2f}, GlobalRate={:.2f}'.format(xm.get_ordinal(), loss.item(), str(i+1), time.asctime(), tracker.rate(), tracker.global_rate() ),flush=True)
        del tok_ids_1, tok_ids_2, flat_maps_1, flat_maps_2, att_masks_1, att_masks_2, weights_1, weights_2, flat_embeddings_1, flat_embeddings_2, embeddings_1, embeddings_2, embeddings_orig_1, embeddings_orig_2
        del loss

    xm.save(model.state_dict(), ckpt)
    gc.collect()





def train(index,dataset,batch_size,model,model_orig,ckpt,seed,epochs,print_every):
    seed_everything(seed)
    device = xm.xla_device()
    
    # if not xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')

    # token_map_dataset = WikipediaTokenMapDataset(files)
    
    # xm.master_print("Loaded Dataset")

    # if xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(token_map_dataset,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
    token_map_dl = DataLoader(token_map_dataset,batch_size=batch_size,sampler=dist_sampler,num_workers=0,drop_last=True, collate_fn=token_maps_collate)
    gc.collect()
    model.to(device)
    model_orig.to(device)
    model_orig.eval()
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
    ]

    optim = AdamW(optimizer_grouped_parameters,lr=5e-5)
    # mse_loss = MSELoss()
    for epoch in range(epochs):
        para_token_map_dl = pl.ParallelLoader(token_map_dl,[device]).per_device_loader(device)
        train_loop(para_token_map_dl,model,optim,device,epoch,ckpt,print_every)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',nargs='+',type=str,help='List of pre-processed token mapping files')
    parser.add_argument('--batch_size',type=int,default=128,help='Train batch size')
    parser.add_argument('--load',type=str,help='Path to pretrained model checkpoint')
    parser.add_argument('--is_tf',action='store_true',default='',help='Whether the model is trained using TF or PyTorch')
    parser.add_argument('--ckpt',type=str,help='Path where model is to be saved')
    parser.add_argument('--bert_config',type=str,help='Path to BERT config file')
    parser.add_argument('--seed',type=int,default=1234,help='Random seed')
    parser.add_argument('--num_epochs',type=int,default=10,help='Number of epochs for training')
    parser.add_argument('--print_every',type=int,default=100)
    args = parser.parse_args()
    token_map_dataset = WikipediaTokenMapDataset(args.files)
    print("Loaded Dataset")
    config = BertConfig.from_json_file(args.bert_config)
    if args.load:
        if args.is_tf:
            model = load_BFTC_from_TF_ckpt(config,args.load,BertModel)
            model_orig = load_BFTC_from_TF_ckpt(config,args.load,BertModel)
        else:
            model = BertModel.from_pretrained(args.load,config=config)
            model_orig = BertModel.from_pretrained(args.load,config=config)
        print("Loaded pretrained model")
    else:
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        model_orig = BertModel.from_pretrained('bert-base-multilingual-cased')
    xmp.spawn(train,args=(token_map_dataset,args.batch_size,model,model_orig,args.ckpt,args.seed,args.num_epochs,args.print_every),nprocs=8,start_method='fork')


        



        



