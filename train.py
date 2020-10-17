import os
import sys
from dataloader import WikipediaTokenMapDataset, token_maps_collate
from transformers import BertModel, AdamW
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



os.environ['XLA_USE_BF16']="1"

def seed_everything(seed):
       # Sets a common random seed - both for initialization and ensuring graph is the same
       random.seed(seed)
       os.environ['PYTHONHASHSEED'] = str(seed) # this should be done before firing Python itself
       np.random.seed(seed)
       torch.manual_seed(seed)

def train_loop(dl, model,model_orig, optimizer, device, epoch, loss_fn):
    model.train()
    xm.master_print("Epoch ",epoch,"/",10)
    for i,(tok_ids_1,tok_ids_2,flat_maps_1,flat_maps_2, att_masks_1, att_masks_2,weights_1, weights_2) in enumerate(dl):
        start_time = time.time()
        embeddings_1 = model(input_ids = tok_ids_1, attention_mask= att_masks_1)[0]
        embeddings_2 = model(input_ids = tok_ids_2, attention_mask= att_masks_2)[0]
        embeddings_orig_1 = model_orig(input_ids = tok_ids_1, attention_mask= att_masks_1)[0]
        embeddings_orig_2 = model_orig(input_ids = tok_ids_2, attention_mask= att_masks_2)[0]
        hidden_size = embeddings_1.size(2)
        flat_embeddings_1 = embeddings_1.view(-1,hidden_size)
        flat_embeddings_2 = embeddings_2.view(-1,hidden_size)
        flat_orig_embeddings_1 = embeddings_orig_1.view(-1,hidden_size)
        flat_orig_embeddings_2 = embeddings_orig_2.view(-1,hidden_size)
        loss = loss_fn(flat_embeddings_1[flat_maps_1]*weights_1,flat_embeddings_2[flat_maps_2]*weights_2)
        loss+= loss_fn(flat_orig_embeddings_1[flat_maps_1]*weights_1,flat_orig_embeddings_2[flat_maps_2]*weights_2)
        # loss = loss_fn(embeddings_1,embeddings_2)
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)
        
        if (i+1)%1 == 0:
            xm.master_print('Iteration: {}, Loss:{}, Time:{}s'.format(str(i+1), loss.item(), time.time()-start_time))
            # xm.save(model.state_dict(),'ckpt/bert_model.hdf5')
    




def train(index,dataset,batch_size,model,model_orig):
    seed_everything(1234)
    device = xm.xla_device()
    
    # if not xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')

    # token_map_dataset = WikipediaTokenMapDataset(files)
    
    # xm.master_print("Loaded Dataset")

    # if xm.is_master_ordinal():
    #     xm.rendezvous('download_only_once')
    
    dist_sampler = torch.utils.data.DistributedSampler(dataset,num_replicas=xm.xrt_world_size(),rank=xm.get_ordinal(),shuffle=True)
    token_map_dl = DataLoader(token_map_dataset,batch_size=batch_size,sampler=dist_sampler,num_workers=0,drop_last=True, collate_fn=token_maps_collate)
    
    model = model.to(device)
    model_orig = model_orig.to(device)
    model_orig.eval()
    optim = AdamW(model.parameters(),lr=5e-5)
    mse_loss = MSELoss()
    iterations=0
    for epoch in range(10):
        para_token_map_dl = pl.ParallelLoader(token_map_dl,[device]).per_device_loader(device)
        train_loop(para_token_map_dl,model,model_orig,optim,device,epoch,mse_loss)
        



if __name__ == "__main__":
    seed_everything(1234)
    filename = sys.argv[1]
    batch_size = int(sys.argv[2])
    token_map_dataset = WikipediaTokenMapDataset(filename)
    print("Loaded Dataset")
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    model_orig = BertModel.from_pretrained('bert-base-multilingual-cased')
    xmp.spawn(train,args=(token_map_dataset,batch_size,model,model_orig),nprocs=8,start_method='fork')

    # train(filename,batch_size,tokenizer,model,model_orig)


        



        



