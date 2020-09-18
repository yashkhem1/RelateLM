import os
import sys
import random
import pickle
import string
from tqdm import tqdm

def create_pseudo_translation_oscar(l1,l2):
    dictionary = pickle.load(open(os.path.join('Bilingual_Mappings',l1+'_'+l2+'_dict.pkl'),'rb'))
    monolingual_file = l1+'_sentences.txt'
    pseudo_translation = l2+'_pseudo_'+l1+'.txt'
    with open(monolingual_file,'r') as f:
        with open(pseudo_translation,'w') as w:
            for i,line in tqdm(enumerate(f)):
                translation = ""
                for word in line.split(" "):
                    punct = ""
                    nl = ""
                    punctuations = ['...']+list(string.punctuation)
                    if word.endswith("\n"):
                        word = word[:-1]
                        nl = "\n"
                    for punctuation in punctuations:
                        if word.endswith(punctuation):
                            word = word[:-1*len(punctuation)]
                            punct = punctuation
                    if word in dictionary.keys():
                        translated = dictionary[word] + punct + nl
                    else:
                        translated = word + punct + nl
                    
                    translation+=translated+" "
                
                w.write(translation[:-1])


if __name__=="__main__":
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    create_pseudo_translation_oscar(l1,l2)