import os
import sys
import random
import pickle
import argparse
from indictrans import Transliterator

def get_bilingual_dictionary(bilingual,o1,o2,l1,l2,transliterate,trans_lang,include_wiktionary,freq_1,freq_2):
    freqs_1 = pickle.load(open(freq_1,'rb'))
    freqs_2 = pickle.load(open(freq_2,'rb'))

    l1_l2_dict = {}
    l2_l1_dict = {}
    trn = None

    if transliterate:
        if trans_lang == 'l1':
            trn = Transliterator(source=l2,target=l1,build_lookup=True)

        elif trans_lang == 'l2':
            trn = Transliterator(source=l1,target=l2,build_lookup=True)

    elif l1 == "hin":
            trn = Transliterator(source=l1,target=l2,build_lookup=True)

    elif l2 == "hin":
            trn = Transliterator(source=l2,target=l1,build_lookup=True)

    freqs = [freqs_2,freqs_1]
    dict_1 = [l1_l2_dict,l2_l1_dict]
    dict_2 = [l2_l1_dict,l1_l2_dict]
    l1_bool = [True,False]
    l2_bool = [False,True]

    for i,dir_ in enumerate(bilingual):
        if not os.path.exists(dir_):
            print("Directory {} not present".format(dir_))
            exit(1)
        else:
            files = os.listdir(dir_)
            if "wiktionary.txt" in files:
                files.remove("wiktionary.txt")

            if trans_lang == "l1" and transliterate:
                populate_dict(dir_,files,dict_1[i],dict_2[i],True,not l1_bool[i],not l2_bool[i],trn,freqs[i])

            elif trans_lang == "l2" and transliterate:
                populate_dict(dir_,files,dict_1[i],dict_2[i],True,l1_bool[i],l2_bool[i],trn,freqs[i])

            else:
                populate_dict(dir_,files,dict_1[i],dict_2[i],True,False,False,trn,freqs[i])

    if include_wiktionary:
        #This will always have one of the languages as Hindi
        #The logic can be made less complicated if "wiktionary.txt" keys can be transliterated to their original script
        #but we don't want any information loss due to transliteration.
        for i,dir_ in enumerate(bilingual):
            files = os.listdir(dir_)
            if "wiktionary.txt" in files:
                files = ["wiktionary.txt"]

                if transliterate:
                    if (trans_lang == "l1" and l1 != "hin") or (trans_lang == "l2" and l2 != "hin"):
                        populate_dict(dir_,files,dict_1[i],dict_2[i],False,True,True,trn,freqs[i])

                    else:
                        populate_dict(dir_,files,dict_1[i],dict_2[i],False,False,False,trn,freqs[i])

                else:
                    if l1 == "hin":
                        populate_dict(dir_,files,dict_1[i],dict_2[i],False,not l1_bool[i], not l2_bool[i],trn,freqs[i])

                    else:
                        populate_dict(dir_,files,dict_1[i],dict_2[i],False,l1_bool[i],l2_bool[i],trn,freqs[i])


    print("Size of",l1,"-",l2,"dictionary:",len(l1_l2_dict.keys()))
    print("Size of",l2,"-",l1,"dictionary:",len(l2_l1_dict.keys()))

    with open(o1,'wb') as w:
        pickle.dump(l1_l2_dict,w)

    with open(o2,'wb') as w:
        pickle.dump(l2_l1_dict,w)

def populate_dict(dir_,files,dict_1,dict_2,can_replace=False,trans_l1=False,trans_l2=False,trn=None,freqs=None):
    for file_ in files:
        with open(os.path.join(dir_,file_),encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            l1_word = line.split(":")[0]
            if trans_l1:
                l1_word = trn.transform(l1_word)

            l2_words = line.split(":")[1][1:]
            if trans_l2:
                l2_words = trn.transform(l2_words)

            l2_words = l2_words.split(" ")
            l2_words = [x.strip() for x in l2_words if x.strip()]
            if l1_word and (len(l2_words) > 0):
                if (l1_word not in dict_1) or can_replace:
                    l1_l2_words = l2_words
                    l1_l2_freqs = [freqs[x]+1 if x in freqs else 1 for x in l2_words]
                    dict_1[l1_word] = [l1_l2_words, l1_l2_freqs]

                for word in l2_words:
                    if(word not in dict_2):
                        dict_2[word] = [[l1_word],[1.0]]



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
    parser.add_argument('--freq_1',type=str,help='Path to frequency file for l1')
    parser.add_argument('--freq_2',type=str,help='Path to frequency file for l2')
    args = parser.parse_args()
    get_bilingual_dictionary(args.bilingual,args.o1,args.o2,args.l1,args.l2,args.transliterate,args.trans_lang,args.include_wiktionary,args.freq_1, args.freq_2)
