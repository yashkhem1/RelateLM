import os
import sys
import random
import pickle
import argparse
from indictrans import Transliterator

def get_bilingual_dictionary(bilingual,o1,o2,l1,l2,transliterate,trans_lang,include_wiktionary,choose_random=True):
    l1_l2_dict = {}
    l2_l1_dict = {}
    if transliterate:
        if trans_lang == 'l1':
            trn = Transliterator(source=l2,target=l1,build_lookup=True)
        else:
            trn = Transliterator(source=l1,target=l2,build_lookup=True)
    
    dir1 = bilingual[0]
    if os.path.exists(dir1):
        files = list(os.listdir(dir1))
        if 'wiktionary.txt' in files: #This is done so that if lexicon is already present in other file, wiktionary doesn't override it, since it is less reliable
            files.remove('wiktionary.txt')
            files.append('wiktionary.txt')
        for file_ in files:
            already_trans = False
            if file_ == 'wiktionary.txt':
                if not include_wiktionary:
                    continue
                else:
                    already_trans = True
            with open(os.path.join(dir1,file_),encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                l1_word = line.split(":")[0]
                if transliterate and trans_lang == 'l2' and not already_trans:
                    l1_word = trn.transform(l1_word)
                l1_word = l1_word.strip()

                l2_words = line.split(":")[1][1:]
                if transliterate and trans_lang == 'l1' and not already_trans:
                    l2_words = trn.transform(l2_words)
                l2_words = l2_words.split(" ")
                if choose_random:
                    l2_word = l2_words[0].strip()
                
                if l1_word and l2_word and l1_word not in l1_l2_dict:
                    l1_l2_dict[l1_word] = l2_word
                
                for word in l2_words:
                    word = word.strip()
                    if word and l1_word and word not in l2_l1_dict:
                        l2_l1_dict[word] = l1_word

    if len(bilingual) > 1:
        dir2 = bilingual[1]
        if os.path.exists(dir2):
            files = list(os.listdir(dir2))
            if 'wiktionary.txt' in files:
                files.remove('wiktionary.txt')
                files.append('wiktionary.txt')
            for file_ in files:
                already_trans = False
                if file_ == 'wiktionary.txt':
                    if not include_wiktionary:
                        continue
                    else:
                        already_trans = True
                with open(os.path.join(dir2,file_),encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    l2_word = line.split(":")[0]
                    if transliterate and trans_lang == 'l1' and not already_trans:
                        l2_word = trn.transform(l2_word)
                    l2_word = l2_word.strip()

                    l1_words = line.split(":")[1][1:]
                    if transliterate and trans_lang == 'l2' and not already_trans:
                        l1_words = trn.transform(l1_words)
                    l1_words = l1_words.split(" ")
                    if choose_random:
                        l1_word = l1_words[0].strip()
                    
                    if l2_word and l1_word and l2_word not in l2_l1_dict:
                        l2_l1_dict[l2_word] = l1_word

                    for word in l1_words:
                        if word not in l1_l2_dict:
                            word = word.strip()
                            if word and l2_word:
                                l1_l2_dict[word] = l2_word
    
    print("Size of",l1,"-",l2,"dictionary:",len(l1_l2_dict.keys()))
    print("Size of",l2,"-",l1,"dictionary:",len(l2_l1_dict.keys()))

    with open(o1,'wb') as w:
        pickle.dump(l1_l2_dict,w)

    with open(o2,'wb') as w:
        pickle.dump(l2_l1_dict,w)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--l1',type=str,help='code for Langugage 1')
    parser.add_argument('--l2',type=str,help='code for Language 2')
    parser.add_argument('--bilingual',nargs='+',type=str,help='Paths to Bilingual Mappings directories')
    parser.add_argument('--o1',type=str,help='Output path to l1-l2 dict')
    parser.add_argument('--o2',type=str,help='Output path to l2-l1 dict')
    parser.add_argument('--transliterate',action='store_true',help='Transliterate to other language')
    parser.add_argument('--trans_lang',type=str,help='Output language script',choices=['l1','l2'],default='l1')
    parser.add_argument('--include_wiktionary',action='store_true',help='Whether to include wiktionary mappings as well')
    args = parser.parse_args()
    get_bilingual_dictionary(args.bilingual,args.o1,args.o2,args.l1,args.l2,args.transliterate,args.trans_lang,args.include_wiktionary)
