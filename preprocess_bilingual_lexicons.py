from transformers import BertTokenizer
import torch
import os
import sys
import string
import pickle
import argparse
import numpy as np

def preprocess_dictionary_files(dict1_file, dict2_file,outfile,max_length,tokenizer):
    with open(dict1_file,'rb') as f:
        dict1 = pickle.load(f)
    with open(dict2_file,'rb') as f:
        dict2 = pickle.load(f)
    with open(outfile,'w') as w:
        _ = w.write('{\n')
        _ = w.write('\t"data":[\n')
        first = 1
        for key in dict1.keys():
            l1_word = key.strip()
            l1_word = l1_word.replace("_"," ")
            l1_toks = tokenizer.encode(l1_word,add_special_tokens=True,pad_to_max_length=True,truncation=True,max_length=max_length)
            l2_word = dict1[key].strip()
            l2_word = l2_word.replace("_"," ")
            l2_toks = tokenizer.encode(l2_word,add_special_tokens=True,pad_to_max_length=True,truncation=True,max_length=max_length)
            if first:
                _ = w.write('\t\t{"token_ids_1":'+str(l1_toks)+',"token_ids_2":'+str(l2_toks)+'}')
                first = 0
            else:
                _ = w.write(',\n\t\t{"token_ids_1":'+str(l1_toks)+',"token_ids_2":'+str(l2_toks)+'}')

        for key in dict2.keys():
            l2_word = key.strip()
            l2_word = l2_word.replace("_"," ")
            l2_toks = tokenizer.encode(l2_word,add_special_tokens=True,pad_to_max_length=True,truncation=True,max_length=max_length)
            l1_word = dict2[key].strip()
            l1_word = l1_word.replace("_"," ")
            l1_toks = tokenizer.encode(l1_word,add_special_tokens=True,pad_to_max_length=True,truncation=True,max_length=max_length)
            _ = w.write(',\n\t\t{"token_ids_1":'+str(l1_toks)+',"token_ids_2":'+str(l2_toks)+'}')
        
        _ = w.write('\n\t]\n')
        _ = w.write('}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict1",type=str,help='Path to dictionary file 1')
    parser.add_argument("--dict2",type=str,help='Path to dictionary file 2')
    parser.add_argument("--vocab_file",type=str,help='Path to vocabulary file')
    parser.add_argument("--max_length",type=int,help="Maximum number of tokens in a sentence")
    parser.add_argument("--outfile",type=str,help="Path to output json file")
    args = parser.parse_args()
    tokenizer = BertTokenizer(vocab_file=args.vocab_file,do_lower_case=False)
    preprocess_dictionary_files(args.dict1,args.dict2,args.outfile,args.max_length,tokenizer)