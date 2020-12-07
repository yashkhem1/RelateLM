import os
import sys
import random
import pickle
import string
from tqdm import tqdm
import progressbar
import argparse
from indictrans import Transliterator
import numpy as np

def create_pseudo_translation(mono,dict_path,outfile,prob=False,transliterate=False,l1='hin',l2='pan'):
    if not os.path.exists(dict_path):
        print("Dictionary file is not present")
        return
    dictionary = pickle.load(open(dict_path,'rb'))

    if not os.path.exists(mono):
        print("Monolingual data doesn't exist")
        return

    if transliterate:
        trn = Transliterator(source=l1,target=l2,build_lookup=True)
    
    with open(mono,'r') as f:
        lines = f.readlines()
    bar = progressbar.ProgressBar(max_value=len(lines),suffix="Number of lines translated: {variables.nlines}",
                                    variables={'nlines':'-'})
    print('Read Complete')
    with open(outfile,'w') as w:
        for i,line in enumerate(lines):
            translation = ""
            for word in line.split(" "):
                punct = ""
                nl = ""
                punctuations = ['...','।']+list(string.punctuation)
                if word.endswith("\n"):
                    word = word[:-1]
                    nl = "\n"
                for punctuation in punctuations:
                    if word.endswith(punctuation):
                        word = word[:-1*len(punctuation)]
                        punct = punctuation
                        break
                if word in dictionary and  word not in ('\n',' ','',' \n'):
                    if prob:
                        translation_words = dictionary[word][0]
                        word_probs = np.array(dictionary[word][1])
                        translated = np.random.choice(translation_words,p=word_probs/word_probs.sum()) + punct + nl
                    else:
                        if isinstance(dictionary[word],list):
                            translated = dictionary[word][0][0] + punct + nl
                        else:
                            translated = dictionary[word] + punct + nl
                else:
                    if transliterate:
                        trans_word = trn.transform(word)
                        if trans_word:
                            translated = trans_word + punct + nl
                        else:
                            translated = word + punct + nl
                    else:
                        translated = word + punct + nl
                
                translation+=translated+" "

            w.write(translation[:-1])
            bar.update(i,nlines=str(i))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mono',type=str,help='Path to monlingual data')
    parser.add_argument('--dict_path',type=str,help='Path to dictionary file')
    parser.add_argument('--outfile',type=str,help='Path to output file')
    parser.add_argument('--prob',action='store_true',help='Probabilistic translation')
    parser.add_argument('--transliterate',action='store_true',help='Transliterate to other script')
    parser.add_argument('--l1',type=str,help='Code for language 1')
    parser.add_argument('--l2',type=str,help='Code for language 2')
    args = parser.parse_args()
    create_pseudo_translation(args.mono,args.dict_path,args.outfile,args.prob,args.transliterate,args.l1,args.l2)