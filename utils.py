from transformers import BertTokenizer
import torch
import os
import sys
import string
import progressbar
import pickle

punctuations = list(string.punctuation)

def tokenize_with_mappings(l1,l2,tokenizer,word_mapping = None):
    l1 = l1.replace("\n","")
    l2 = l2.replace("\n","")
    token_ids_l1 = []
    token_ids_l2 = []
    token_maps_l1 = []
    token_maps_l2 = []
    if word_mapping is None:
        words_l1 = l1.split(" ")
        words_l2 = l2.split(" ")
        index_l1 = 0
        index_l2 = 0 

        assert(len(words_l1) == len(words_l2))
        for i in range(len(words_l1)):
            word_l1 = words_l1[i].replace("_"," ")
            word_l2 = words_l2[i].replace("_"," ")
            tokens_1 = tokenizer.encode(word_l1,add_special_tokens = False)
            tokens_2 = tokenizer.encode(word_l2,add_special_tokens = False)
            token_ids_l1 += tokens_1
            token_ids_l2 += tokens_2
            token_maps_l1.append(index_l1+len(tokens_1)-1)
            token_maps_l2.append(index_l2+len(tokens_2)-1)
            index_l1 += len(tokens_1)
            index_l2 += len(tokens_2)
    return token_ids_l1, token_ids_l2, token_maps_l1, token_maps_l2


def preprocess_with_mappings(file1,file2,outfile,max_length,tokenizer,document_information=True,word_mapping = None):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    with open(file1,'r') as f1:
        with open(file2,'r') as f2:
            with open(outfile,'a') as w:
                _ = w.write('{\n')
                _ = w.write('\t"data":[\n')
                line1 = f1.readline()
                line2 = f2.readline()
                i=0
                first = 1
                ids_1 = [101]
                ids_2 = [101]
                tok_maps_1 = []
                tok_maps_2 = []
                while line1:
                    if document_information:
                        token_ids_l1, token_ids_l2, token_maps_l1, token_maps_l2 = tokenize_with_mappings(line1,line2,tokenizer,word_mapping)
                        if (len(ids_1) + len(token_ids_l1) > max_length) or (len(ids_2) + len(token_ids_l2) > max_length) or line1=="\n":
                            if not (len(ids_1)==1 or len(ids_2)==1):
                                ids_1 += [102]+[0]*(max_length-len(ids_1)-1)
                                ids_2 += [102]+[0]*(max_length-len(ids_2)-1)
                                tok_maps_1 += [0]*(max_length-len(tok_maps_1))
                                tok_maps_2 += [0]*(max_length-len(tok_maps_2))
                                if first:
                                    _ = w.write('\t\t{"token_ids_1":'+str(ids_1)+',"token_ids_2":'+str(ids_2)+',"token_maps_1":'+str(tok_maps_1)+',"token_maps_2":'+str(tok_maps_2)+'}')
                                    first = 0
                                
                                else:
                                    _ = w.write(',\n\t\t{"token_ids_1":'+str(ids_1)+',"token_ids_2":'+str(ids_2)+',"token_maps_1":'+str(tok_maps_1)+',"token_maps_2":'+str(tok_maps_2)+'}')

                            if line1=="\n" or len(ids_1)==1 or len(ids_2)==1:
                                line1 = f1.readline()
                                line2 = f2.readline()
                                bar.update(i)
                                i+=1
                            
                            ids_1 = [101]
                            ids_2 = [101]
                            tok_maps_1 = []
                            tok_maps_2 = []
                        
                        else:
                            tok_maps_1 += [x+len(ids_1) for x in token_maps_l1]
                            tok_maps_2 += [x+len(ids_2) for x in token_maps_l2]
                            ids_1 += token_ids_l1
                            ids_2 += token_ids_l2
                            line1 = f1.readline()
                            line2 = f2.readline()
                            bar.update(i)
                            i+=1
    
                _ = w.write('\n\t]\n')
                _ = w.write('}')


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # l1 = sys.argv[1]
    # l2 = sys.argv[2]
    # get_token_mapping_file(l1+'_sentences.txt',l2+'_pseudo_'+l1+'.txt','token_map.txt')
    # get_token_mapping_file(l2+'_sentences.txt',l1+'_pseudo_'+l2+'.txt','token_map.txt')
    # get_token_mapping_file('original_hindi.txt','pseudo_marathi.txt','token_map.txt')
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    max_length = int(sys.argv[3])
    corpus = sys.argv[4]
    if corpus == 'Wikipedia':
        file1 = os.path.join('Wikipedia',l1+'.txt')
        file2 = os.path.join('Wikipedia',l2+'_pseudo_'+l1+'.txt')
        outfile = l1 + '_' + l2 +'.json'
        preprocess_with_mappings(file1,file2,outfile,max_length,tokenizer)
                    




    
