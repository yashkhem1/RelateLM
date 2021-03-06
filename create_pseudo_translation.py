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
from create_normalised_dict import normaliser
from tokenizers import BertWordPieceTokenizer

def create_pseudo_translation(mono,dict_path,outfile,replace='first',transliterate=False,l1='hin',l2='pan', norm=False, normalised_dict_path=None, tok=False, tokenizer=None):
    if not os.path.exists(dict_path):
        print("Dictionary file is not present")
        return
    dictionary = pickle.load(open(dict_path,'rb'))

    normalised_dictionary = None
    if norm:
        if not os.path.exists(normalised_dict_path):
            print("Normalised Dictionary file is not present")
            return
        normalised_dictionary = pickle.load(open(normalised_dict_path,'rb'))

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
            if tok:
                line = line[:-1]
                line = (" ".join(tokenizer.encode(line.replace(" "," *#* "),add_special_tokens = False).tokens)).replace('##'," ## ")
                line += '\n'

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

                translation_present = False
                normalised_word = normaliser(word)

                if word in dictionary and  word not in ('\n',' ','',' \n'):
                    translation_present = True
                    translation_key = word
                    translation_dict = dictionary

                elif norm and normalised_word in dictionary and normalised_word not in ('\n',' ','','\n'):
                    translation_present = True
                    translation_key = normalised_word
                    translation_dict = dictionary

                elif norm and normalised_word in normalised_dictionary and normalised_word not in ('\n',' ','','\n'):
                    translation_present = True
                    translation_key = normalised_word
                    translation_dict = normalised_dictionary

                if translation_present:
                    translation_words = translation_dict[translation_key][0]
                    word_probs = np.array(translation_dict[translation_key][1])

                    if replace=='prob':
                        translated = np.random.choice(translation_words,p=np.sqrt(word_probs)/np.sqrt(word_probs).sum()) + punct + nl

                    elif replace=='problin':
                        translated = np.random.choice(translation_words,p=word_probs/word_probs.sum()) + punct + nl

                    elif replace=='first':
                        translated = translation_words[0] + punct + nl

                    elif replace=='max':
                        translated = translation_words[np.argmax(word_probs)] + punct + nl

                    elif replace=='random':
                        translated = np.random.choice(translation_words) + punct + nl

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

            translation = translation[:-1]
            if tok:
                translation = translation.replace(" ","").replace("##", "").replace("*#*"," ")
            w.write(translation)
            bar.update(i,nlines=str(i))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mono',type=str,help='Path to monlingual data')
    parser.add_argument('--dict_path',type=str,help='Path to dictionary file')
    parser.add_argument('--outfile',type=str,help='Path to output file')
    parser.add_argument('--replace',type=str,help='Method to replace dictionary matches',choices=['first','prob','max','random','problin'])
    parser.add_argument('--transliterate',action='store_true',help='Transliterate to other script')
    parser.add_argument('--l1',type=str,help='Code for language 1')
    parser.add_argument('--l2',type=str,help='Code for language 2')
    parser.add_argument('--norm',action='store_true',help='Normalization of words')
    parser.add_argument('--norm_dict_path',type=str,help='Path to normalized dictionary file')
    parser.add_argument('--tok',action='store_true',help='Tokenization of words')
    parser.add_argument('--tok_vocab_path',type=str,help='Path to tokenization vocab file')
    args = parser.parse_args()
    if args.tok:
        tokenizer = BertWordPieceTokenizer(args.tok_vocab_path,lowercase=False,strip_accents=False)
    else:
        tokenizer = None
    create_pseudo_translation(args.mono,args.dict_path,args.outfile,args.replace,args.transliterate,args.l1,args.l2, args.norm, args.norm_dict_path, args.tok, tokenizer)