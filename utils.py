from transformers import BertTokenizer,BertTokenizerFast
import torch
import os
import sys
import string
import progressbar

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
punctuations = list(string.punctuation)

def get_token_mappings_line(l1,l2,word_mapping = None):
    l1 = l1.replace("\n","")
    l2 = l2.replace("\n","")
    token_mapping = []
    index_l1 = 1
    index_l2 = 1
    if word_mapping is None:
        words_l1 = l1.split(" ")
        words_l2 = l2.split(" ")
        assert(len(words_l1) == len(words_l2))
        for i in range(len(words_l1)):
            word_l1 = words_l1[i].replace("_"," ")
            word_l2 = words_l2[i].replace("_"," ")
            tokens_1 = tokenizer(word_l1, add_special_tokens=False)
            tokens_2 = tokenizer(word_l2, add_special_tokens=False)
            token_mapping.append((index_l1+len(tokens_1)-1,index_l2+len(tokens_2)-1))
            index_l1 += len(tokens_1)
            index_l2 += len(tokens_2)

    return token_mapping




def get_token_mapping_file(file1,file2,tok_file):
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    with open(file1,'r') as f1:
        with open(file2,'r') as f2:
            with open(tok_file,'a') as f3:
                line1 = f1.readline()
                line2 = f2.readline()
                i=0
                while line1:
                    map_string = ""
                    _,_,token_mapping = get_token_mappings_line(line1,line2)
                    for mapping in token_mapping:
                        map_string+="("+str(mapping[0])+","+str(mapping[1])+")"+";"
                    map_string = map_string[:-1]+"\n"
                    f3.write(map_string)
                    line1 = f1.readline()
                    line2 = f2.readline()
                    bar.update(i)
                    i+=1


if __name__ == "__main__":
    # l1 = sys.argv[1]
    # l2 = sys.argv[2]
    # get_token_mapping_file(l1+'_sentences.txt',l2+'_pseudo_'+l1+'.txt','token_map.txt')
    # get_token_mapping_file(l2+'_sentences.txt',l1+'_pseudo_'+l2+'.txt','token_map.txt')
    get_token_mapping_file('original_hindi.txt','pseudo_marathi.txt','token_map.txt')
                    




    
