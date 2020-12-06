import os
import sys
import random
import progressbar
import argparse
import pickle
import string


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mono',type=str,help='Path to monlingual data')
    parser.add_argument('--dict_path',type=str,help='Path to dictionary file')
    parser.add_argument('--translation',type=str,help='Path to translation file')
    parser.add_argument('--stats',type=str,help='Statistics type')
    parser.add_argument('--outfile',type=str,help='Path to output file')
    parser.add_argument('--freq_file',type=str,help='Path to file containing frequency dictionary')
    parser.add_argument('--k',type=int,help='Top k elements')

    args = parser.parse_args()
    if args.stats == 'pseudo_translation':
        get_pseudo_translation_statistics(args.mono,args.dict_path,args.translation)
    elif args.stats == 'word_freq':
        word_frequencies(args.mono,args.outfile)
    elif args.stats == 'top_k_translated':
        top_k_translation(args.freq_file,args.dict_path,args.outfile,args.k,True)
    elif args.stats == 'top_k_not_translated':
        top_k_translation(args.freq_file,args.dict_path,args.outfile,args.k,False)
    
    

