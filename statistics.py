import os
import sys
import random
import progressbar
import argparse
import pickle
import string
from transformers import BertModel, BertConfig
from prettytable import PrettyTable
import numpy as np


def get_pseudo_translation_statistics(mono,dict_path,outfile):
    dictionary = pickle.load(open(dict_path,'rb'))
    dict_keys = list(dictionary.keys())
    dict_keys_used = set()
    total_words = 0
    words_translated = 0
    i = 0
    with open(mono,'r') as f1:
        with open(outfile,'r') as f2:
            lines_l1 = f1.readlines()
            lines_l2 = f2.readlines()
    print("Read Complete")
    assert(len(lines_l1)==len(lines_l2))
    bar = progressbar.ProgressBar(term_width=20,max_value=len(lines_l1),suffix="Total Dict: {variables.total_dict} Used Dict: {variables.dict_used} Used%: {variables.used_perc} Total: {variables.total_words} Translated: {variables.translated} Translated%: {variables.translated_perc}",
                                                                        variables = {'total_dict':'--','dict_used':'--','used_perc':'--','total_words':'--','translated':'--', 'translated_perc':'--'})
    for (line_l1,line_l2) in zip(lines_l1,lines_l2):
        words_l1 = line_l1.split(" ")
        words_l2 = line_l2.split(" ")
        assert(len(words_l1)==len(words_l2))
        for (word_l1,word_l2) in zip(words_l1,words_l2):
            total_words += 1
            word_l1 = word_l1.strip()
            word_l2 = word_l2.strip()
            if word_l1 != word_l2 and word_l1:
                words_translated+=1
                if word_l1.endswith('\n'):
                    word_l1 = word_l1[:-1]
                punctuations = ['...','।']+list(string.punctuation)
                for punctuation in punctuations:
                    if word_l1.endswith(punctuation):
                        word_l1 = word_l1[:-1*len(punctuation)]
                        break
                if word_l1 not in ('\n',' ','',' \n'):
                    dict_keys_used.add(word_l1)
        # print(len(dict_keys_used))
        i+=1 
        bar.update(i,total_dict=str(len(dict_keys)),dict_used=str(len(dict_keys_used)),used_perc=str(len(dict_keys_used)/len(dict_keys)),
                        total_words=str(total_words),translated=str(words_translated), translated_perc=str(words_translated/total_words))

def word_frequencies(mono,outfile):
    with open(mono,'r') as f:
        lines = f.readlines()
    print("Read Complete")
    freq_dict = {}
    i=0
    unique_words = 0
    bar = progressbar.ProgressBar(max_value=len(lines),suffix="Unique Words encountered : {variables.unique_words}",variables={"unique_words":'--'})
    for line in lines:
        words = line.split(" ")
        for word in words:
            word = word.strip()
            punctuations = ['...','।']+list(string.punctuation)
            for punctuation in punctuations:
                if word.endswith(punctuation):
                    word = word[:-1*len(punctuation)]
                    break
            word = word.strip()
            if word:
                if word in freq_dict:
                    freq_dict[word] += 1
                else:
                    unique_words+=1
                    freq_dict[word] = 0
        
        i+=1
        bar.update(i,unique_words=unique_words)
    
    with open(outfile,'wb') as w:
        pickle.dump(freq_dict,w)

def top_k_translation(freq_file,dict_path,outfile,k=1000,translated=False):
    with open(freq_file,'rb') as f:
        freq_dict = pickle.load(f)
    with open(dict_path,'rb') as f:
        dictionary = pickle.load(f)
    word_freqs = [[x,freq_dict[x]] for x in freq_dict]
    word_freqs.sort(key=lambda x: x[1],reverse=True)
    i = 0
    with open(outfile,'w') as w:
        for (word,freq) in word_freqs:
            if (translated and (word in dictionary)) or (not translated and (word not in dictionary)):
                if translated:
                    w.write(word+":"+dictionary[word]+':'+str(freq)+"\n")
                else:
                    w.write(word+':'+str(freq)+"\n")
                i+=1
                if i == k:
                    break

def display_parameters(model_path, bert_config):
    config = BertConfig.from_json_file(bert_config)
    model = BertModel.from_pretrained(model_path,config=config)
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params+=param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    count_parameters(model)

def compare_transliteration_translation(dict1_path,dict2_path,outfile,choose='first'):
    same_words = set()
    with open(dict1_path,'rb') as f:
        dict1 = pickle.load(f)
    with open(dict2_path,'rb') as f:
        dict2 = pickle.load(f)
    for dict_ in [dict1,dict2]:
        for key in dict_:
            translation_words = dict_[key][0]
            word_probs = np.array(dict_[key][1])
            if choose=="first":
                translation = translation_words[0]
            elif choose=="max":
                translation = translation_words[np.argmax(word_probs)]
            elif choose=="any":
                translation = key if key in translation_words else translation_words[0]
                
            if  key.strip() and translation.strip()==key.strip():
                same_words.add(key)

    print("Number of words for which translation is same as transliteration:",len(same_words))
    with open(outfile,'w') as w:
        for word in same_words:
            w.write(word+'\n')

def compare_vocab_freq(vocab_path,freq_files,outfile,start_index,end_index):
    whole_words = set()
    with open(vocab_path,'r') as f:
        vocab = f.readlines()
    vocab = [x.strip() for x in vocab]
    if start_index!=-1:
        vocab = vocab[start_index:]
    if end_index!=-1:
        vocab = vocab[:end_index]
    for freq_file in freq_files:
        print("Frequency File:",freq_file)
        with open(freq_file,'rb') as f:
            freqs = pickle.load(f)
        for word in vocab:
            if word in freqs:
                whole_words.add(word)
    print("Number of whole words in vocabulary:",len(whole_words))
    with open(outfile,'w') as w:
        for word in whole_words:
            w.write(word+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mono',type=str,help='Path to monlingual data')
    parser.add_argument('--dict_path',type=str,help='Path to dictionary file')
    parser.add_argument('--translation',type=str,help='Path to translation file')
    parser.add_argument('--stats',type=str,help='Statistics type')
    parser.add_argument('--outfile',type=str,help='Path to output file')
    parser.add_argument('--freq_files',type=str,nargs='+',help='Path to file containing frequency dictionary')
    parser.add_argument('--k',type=int,help='Top k elements')
    parser.add_argument('--model_path',type=str,help='Path to saved model')
    parser.add_argument('--bert_config',type=str,help='Path to BERT config')
    parser.add_argument('--dict2_path',type=str,help='Path to dictionary 2')
    parser.add_argument('--choose',type=str,help='Method to choose dictionary translation')
    parser.add_argument('--vocab_file',type=str,help='Path to vocab file')
    parser.add_argument('--vocab_start_index',type=int,default=-1,help='Starting index of vocaublary')
    parser.add_argument('--vocab_end_index',type=int,default=-1,help='End index of vocabulary')

    args = parser.parse_args()
    if args.stats == 'pseudo_translation':
        get_pseudo_translation_statistics(args.mono,args.dict_path,args.translation)
    elif args.stats == 'word_freq':
        word_frequencies(args.mono,args.outfile)
    elif args.stats == 'top_k_translated':
        top_k_translation(args.freq_files[0],args.dict_path,args.outfile,args.k,True)
    elif args.stats == 'top_k_not_translated':
        top_k_translation(args.freq_files[0],args.dict_path,args.outfile,args.k,False)
    elif args.stats == 'disp_params':
        display_parameters(args.model_path,args.bert_config)
    elif args.stats == 'compare_trans':
        compare_transliteration_translation(args.dict_path,args.dict2_path,args.outfile,args.choose)
    elif args.stats == 'whole_words_vocab':
        compare_vocab_freq(args.vocab_file,args.freq_files,args.outfile,args.vocab_start_index,args.vocab_end_index)
    
    

