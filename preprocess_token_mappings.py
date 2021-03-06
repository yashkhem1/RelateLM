from tokenizers import BertWordPieceTokenizer
import torch
import os
import sys
import string
import progressbar
import pickle
import argparse
import numpy as np

punctuations = list(string.punctuation)

def tokenize_with_mappings(l1,l2,tokenizer,word_mapping = None):
    token_ids_l1 = []
    token_ids_l2 = []
    token_maps_l1 = []
    token_maps_l2 = []
    if word_mapping is None:
        words_l1 = l1.split(" ")
        words_l2 = l2.split(" ")
        index_l1 = 0
        index_l2 = 0
        if (len(words_l1) != len(words_l2)):
            print(words_l1)
            print(words_l2)
            raise AssertionError
        for i in range(len(words_l1)):
            word_l1 = words_l1[i].replace("_"," ")
            word_l2 = words_l2[i].replace("_"," ")
            tokens_1 = tokenizer.encode(word_l1,add_special_tokens = False).ids
            tokens_2 = tokenizer.encode(word_l2,add_special_tokens = False).ids
            if len(tokens_1) == 0 or len(tokens_2) == 0:
                continue
            token_ids_l1 += tokens_1
            token_ids_l2 += tokens_2
            token_maps_l1.append(index_l1+len(tokens_1)-1)
            token_maps_l2.append(index_l2+len(tokens_2)-1)
            index_l1 += len(tokens_1)
            index_l2 += len(tokens_2)
    return token_ids_l1, token_ids_l2, token_maps_l1, token_maps_l2


def preprocess_with_mappings(file1,file2,outfile,max_length,tokenizer,cls_id,sep_id,document_information=True,word_mapping = None):
    with open(file1,'r') as f1:
        with open(file2,'r') as f2:
            lines_1 = f1.readlines()
            lines_2 = f2.readlines()
    assert(len(lines_1)==len(lines_2))
    print("Read Complete")
    bar = progressbar.ProgressBar(max_value=len(lines_1))
    with open(outfile,'w') as w:
        _ = w.write('{\n')
        _ = w.write('\t"data":[\n')
        lines_zip = zip(lines_1,lines_2)
        pair = next(lines_zip,None)
        i=0
        first = 1
        ids_1 = [cls_id]
        ids_2 = [cls_id]
        tok_maps_1 = []
        tok_maps_2 = []
        while pair:
            line1,line2 = pair
            if document_information:
                token_ids_l1, token_ids_l2, token_maps_l1, token_maps_l2 = tokenize_with_mappings(line1,line2,tokenizer,word_mapping)
                assert(len(token_maps_l1)==len(token_maps_l2))
                if (len(ids_1) + len(token_ids_l1) >= max_length) or (len(ids_2) + len(token_ids_l2) >= max_length) or not line1.strip():
                    if not (len(ids_1)==1 or len(ids_2)==1):
                        ids_1 += [sep_id]+[0]*(max_length-len(ids_1)-1)
                        ids_2 += [sep_id]+[0]*(max_length-len(ids_2)-1)
                        tok_maps_1 += [0]*(max_length-len(tok_maps_1))
                        tok_maps_2 += [0]*(max_length-len(tok_maps_2))
                        if first:
                            _ = w.write('\t\t{"token_ids_1":'+str(ids_1)+',"token_ids_2":'+str(ids_2)+',"token_maps_1":'+str(tok_maps_1)+',"token_maps_2":'+str(tok_maps_2)+'}')
                            first = 0

                        else:
                            _ = w.write(',\n\t\t{"token_ids_1":'+str(ids_1)+',"token_ids_2":'+str(ids_2)+',"token_maps_1":'+str(tok_maps_1)+',"token_maps_2":'+str(tok_maps_2)+'}')

                    elif  line1.strip():
                        token_ids_l1 = token_ids_l1[:(max_length-2)]
                        token_ids_l2 = token_ids_l2[:(max_length-2)]
                        token_maps_l1 = np.array(token_maps_l1)
                        token_maps_l2 = np.array(token_maps_l2)
                        if len(token_maps_l1[token_maps_l1< (max_length-2)]) < len(token_maps_l2[token_maps_l2 < (max_length-2)]):
                            bool_tok = token_maps_l1 < (max_length-2)
                        else:
                            bool_tok = token_maps_l2 < (max_length-2)
                        token_maps_l1 = token_maps_l1[bool_tok]
                        token_maps_l2 = token_maps_l2[bool_tok]
                        tok_maps_1 += [x+len(ids_1) for x in token_maps_l1]
                        tok_maps_2 += [x+len(ids_2) for x in token_maps_l2]
                        ids_1 += token_ids_l1
                        ids_2 += token_ids_l2
                        pair = next(lines_zip,None)
                        bar.update(i)
                        i+=1
                        continue

                    else:
                        pair = next(lines_zip,None)
                        bar.update(i)
                        i+=1

                    ids_1 = [cls_id]
                    ids_2 = [cls_id]
                    tok_maps_1 = []
                    tok_maps_2 = []

                else:
                    tok_maps_1 += [x+len(ids_1) for x in token_maps_l1]
                    tok_maps_2 += [x+len(ids_2) for x in token_maps_l2]
                    ids_1 += token_ids_l1
                    ids_2 += token_ids_l2
                    pair = next(lines_zip,None)
                    bar.update(i)
                    i+=1

        _ = w.write('\n\t]\n')
        _ = w.write('}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono",type=str,help='Path to monolingual data')
    parser.add_argument("--translated",type=str,help='Path to translated data')
    parser.add_argument("--vocab_file",type=str,help='Path to vocabulary file')
    parser.add_argument("--max_length",type=int,help="Maximum number of tokens in a sentence")
    parser.add_argument("--outfile",type=str,help="Path to output json file")
    args = parser.parse_args()
    tokenizer = BertWordPieceTokenizer(args.vocab_file,lowercase=False,strip_accents=False)
    cls_id = tokenizer.encode("[CLS]",add_special_tokens=False).ids[0]
    sep_id = tokenizer.encode("[SEP]",add_special_tokens=False).ids[0]
    print("CLS_ID:",cls_id,"SEP_ID:",sep_id)
    preprocess_with_mappings(args.mono,args.translated,args.outfile,args.max_length,tokenizer,cls_id,sep_id)
