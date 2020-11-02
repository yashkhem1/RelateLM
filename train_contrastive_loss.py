import os
import sys
from dataloader import BilingualWithNegativeSampling
from transformers import BertModel, AdamW, BertConfig
from utils import load_BFTC_from_TF_ckpt
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch.nn.functional as F
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

def train_loop(dl, model, optimizer, device, epoch,ckpt,print_every,neg_samples,margin,loss_type):
    tracker = xm.RateTracker()
    model.train()
    xm.master_print("Epoch ",epoch,"/",10)
    loss_fn = MSELoss()
    for i,batch in enumerate(dl):
        optimizer.zero_grad()
        tok_ids_1 = batch[0]
        tok_ids_2 = batch[1]
        batch_size = tok_ids_1.shape[0]
        tracker.add(batch_size)
        att_masks_1 = (tok_ids_1>0).to(device,dtype=torch.long)
        att_masks_2 = (tok_ids_2>0).to(device,dtype=torch.long)
        emb_cls_1 = model(input_ids=tok_ids_1,attention_mask=att_masks_1)[0][:,0,:]
        emb_cls_2 = model(input_ids=tok_ids_2,attention_mask=att_masks_2)[0][:,0,:]
        if loss_type == 'exp':
            l_num = torch.exp(F.cosine_similarity(emb_cls_1,emb_cls_2,dim=1)-margin)
            l_den = 0
            l_den_ = 0
        elif loss_type == 'mse':
            l_near = torch.mean((1-F.cosine_similarity(emb_cls_1,emb_cls_2,dim=1)))/(neg_samples+1)
            l_far = 0
        for j in range(neg_samples):
            tok_ids_neg_1 = batch[2*(j+1)]
            tok_ids_neg_2 = batch[2*(j+1)+1]
            att_masks_neg_1 = (tok_ids_neg_1>0).to(device,dtype=torch.long)
            att_masks_neg_2 = (tok_ids_neg_2>0).to(device,dtype=torch.long)
            emb_cls_neg_1 = model(input_ids=tok_ids_neg_1,attention_mask=att_masks_neg_1)[0][:,0,:]
            emb_cls_neg_2 = model(input_ids=tok_ids_neg_2,attention_mask=att_masks_neg_2)[0][:,0,:]
            if loss_type == 'exp':
                l_den += torch.exp(F.cosine_similarity(emb_cls_1,emb_cls_neg_2,dim=1))
                l_den_ += torch.exp(F.cosine_similarity(emb_cls_2,emb_cls_neg_1,dim=1))
            elif loss_type == 'mse':
                l_far += torch.mean(F.relu(F.cosine_similarity(emb_cls_1,emb_cls_neg_2,dim=1)-margin))/(2*(neg_samples+1))
                l_far += torch.mean(F.relu(F.cosine_similarity(emb_cls_2,emb_cls_neg_1,dim=1)-margin))/(2*(neg_samples+1))

        if loss_type == 'exp':
            ls = -1.0*torch.mean((1.0/(neg_samples+1))*l_num/(l_num+l_den))
            ls_ = -1.0*torch.mean((1.0/(neg_samples+1))*l_num/(l_num+l_den_))
            loss = ls + ls_
        elif loss_type == 'mse':
            loss = l_near+ l_far

        loss.backward()
        xm.optimizer_step(optimizer)
        
        if (i+1)%print_every == 0:
            if loss_type == 'exp':
                xm.master_print('[xla:{}], Loss: {}, Iteration: {}, Time:{}s, Rate={:.2f}, GlobalRate={:.2f}'.format(xm.get_ordinal(), loss.item(), str(i+1), time.asctime(), tracker.rate(), tracker.global_rate() ),flush=True)
            
            elif loss_type == 'mse':
                xm.master_print('[xla:{}], Loss: {}, Loss_near: {}, Loss_far: {} Iteration: {}, Time:{}s, Rate={:.2f}, GlobalRate={:.2f}'.format(xm.get_ordinal(), loss.item(), l_near.item(), l_far.item(), str(i+1), time.asctime(), tracker.rate(), tracker.global_rate() ),flush=True)

        del tok_ids_1,tok_ids_2,att_masks_1,att_masks_2,emb_cls_1,emb_cls_2,tok_ids_neg_1,tok_ids_neg_2,att_masks_neg_1,att_masks_neg_2,emb_cls_neg_1,emb_cls_neg_2
        if loss_type == 'exp':
            del loss, ls, ls_,l_num,l_den,l_den_
        elif loss_type == 'mse':
            del loss, l_near, l_far
    
    xm.save(model.state_dict(), ckpt)
    gc.collect()


def train(index,dataset,batch_size,model,ckpt,seed,epochs,print_every,neg_samples,margin,loss_type):
    seed_everything(seed)
    device = xm.xla_device()
    
    # if not xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')

    # token_map_dataset = WikipediaTokenMapDataset(files)
    
    # xm.master_print("Loaded Dataset")

    # if xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
    token_map_dl = DataLoader(dataset,batch_size=batch_size,sampler=dist_sampler,num_workers=0,drop_last=True)
    gc.collect()
    model.to(device)
    
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
        train_loop(para_token_map_dl,model,optim,device,epoch,ckpt,print_every, neg_samples, margin, loss_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files',nargs='+',type=str,help='List of pre-processed dictionary files')
    parser.add_argument('--batch_size',type=int,default=64,help='Train batch size')
    parser.add_argument('--load',type=str,help='Path to pretrained model checkpoint')
    parser.add_argument('--is_tf',action='store_true',default='',help='Whether the model is trained using TF or PyTorch')
    parser.add_argument('--ckpt',type=str,help='Path where model is to be saved')
    parser.add_argument('--bert_config',type=str,help='Path to BERT config file')
    parser.add_argument('--seed',type=int,default=1234,help='Random seed')
    parser.add_argument('--num_epochs',type=int,default=10,help='Number of epochs for training')
    parser.add_argument('--print_every',type=int,default=100)
    parser.add_argument('--margin',type=float,default=0.3,help='Margin for contrastive loss')
    parser.add_argument('--neg_samples',type=int,default=1,help='Number of negative samples')
    parser.add_argument('--loss_type',type=str,default='exp',choices=['mse','exp'],help='Loss type')
    args = parser.parse_args()
    bilingual_dataset = BilingualWithNegativeSampling(args.files,args.neg_samples)
    print("Loaded Dataset")
    config = BertConfig.from_json_file(args.bert_config)
    if args.load:
        if args.is_tf:
            model = load_BFTC_from_TF_ckpt(config,args.load,BertModel)
        else:
            model = BertModel.from_pretrained(args.load,config=config)
        print("Loaded pretrained model")
    else:
        model = BertModel.from_pretrained('bert-base-multilingual-cased')

    xmp.spawn(train,args=(bilingual_dataset,args.batch_size,model,args.ckpt,args.seed,args.num_epochs,args.print_every,args.neg_samples,args.margin,args.loss_type),nprocs=8,start_method='fork')