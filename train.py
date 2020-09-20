import os
import sys
from dataloader import PseudoTokenMappingDataset, custom_collate_fn
from transformers import BertModel, BertTokenizer, AdamW
import torch
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import progressbar

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')

def train(l1,l2):
    token_mapping_datset_1 = PseudoTokenMappingDataset(l1+'_sentences.txt',l2+'_pseudo_'+l1+'.txt')
    token_mapping_datset_2 = PseudoTokenMappingDataset(l2+'_sentences.txt',l1+'_pseudo_'+l2+'.txt')
    token_mapping_dl_1 = DataLoader(token_mapping_datset_1,batch_size=64,collate_fn=custom_collate_fn)
    token_mapping_dl_2 = DataLoader(token_mapping_datset_2,batch_size=64,collate_fn=custom_collate_fn) 
    model = BertModel.from_pretrained('bert-base-multilingual-cased',return_dict=True)
    model_orig = BertModel.from_pretrained('bert-base-multilingual-cased',return_dict=True)
    model = model.to(device)
    model_orig = model_orig.to(device)
    model.train()
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    optim = AdamW(model.parameters(),lr=5e-5)
    iterations = 0
    for epoch in range(10):
        print("Epoch ",epoch,"/",10)
        bar = progressbar.ProgressBar(max_value =progressbar.UnknownLength)
        for dataloader in [token_mapping_dl_1,token_mapping_dl_2]:
            for i,(sent1,sent2,tok_maps) in enumerate(dataloader):
                enc_1 = tokenizer(sent1,padding=True,truncation=True,return_tensors='pt',max_length=512)
                enc_2 = tokenizer(sent2,padding=True,truncation=True,return_tensors='pt',max_length=512)
                embeddings_1 = model(**enc_1).last_hidden_state
                embeddings_2 = model(**enc_2).last_hidden_state
                embeddings_orig_1 = model_orig(**enc_1).last_hidden_state
                embeddings_orig_2 = model_orig(**enc_2).last_hidden_state
                loss = 0
                for j,tok_map in enumerate(tok_maps):
                    emb_1 = embeddings_1[j]
                    emb_orig_1 = embeddings_orig_1[j]
                    emb_2 = embeddings_2[j]
                    emb_orig_2 = embeddings_orig_2[j]
                    max_length_1 = emb_1.shape[1]
                    max_length_2 = emb_2.shape[1] 
                    tokens_1 = [x[0] for x in tok_map if x[0] < max_length_1]
                    tokens_2 = [x[1] for x in tok_map if x[1] < max_length_2]
                    min_toks = min(len(tokens_1,tokens_2))
                    loss += MSELoss(emb_1[tokens_1[:min_toks],:],emb_2[tokens_2[:min_toks],:])
                    loss += MSELoss(emb_1,emb_orig_1)
                    loss += MSELoss(emb_2,emb_orig_2)
                    loss.backward()
                    optim.step()
                bar.suffix = "Loss: "+loss
                bar.update(i)
                iterations +=1
                if iterations%100 == 0:
                        torch.save(model.state_dict(),'ckpt/bert_model.hdf5')
            bar.finish()


if __name__ == "__main__":
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    train(l1,l2)


        



        



