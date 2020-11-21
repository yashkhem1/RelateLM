import os
import sys
import random
import pickle
import string
from tqdm import tqdm
import progressbar
import argparse

def create_pseudo_translation(mono,dict_path,outfile):
    if not os.path.exists(dict_path):
        print("Dictionary file is not present")
        return
    dictionary = pickle.load(open(dict_path,'rb'))

    if not os.path.exists(mono):
        print("Monolingual data doesn't exist")
        return
    
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
                if word in dictionary.keys() and  word not in ('\n',' ','',' \n'):
                    translated = dictionary[word] + punct + nl
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
    args = parser.parse_args()
    create_pseudo_translation(args.mono,args.dict_path,args.outfile)