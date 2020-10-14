import os
import sys
import random
import pickle
import string
from tqdm import tqdm
import progressbar

def create_pseudo_translation(l1,l2,dir_='Wikipedia'):
    dictionary = pickle.load(open(os.path.join('Bilingual_Mappings',l1+'_'+l2+'_dict.pkl'),'rb'))
    monolingual_file = os.path.join(dir_,l1+'.txt')
    pseudo_translation = os.path.join(dir_,l2+'_pseudo_'+l1+'.txt')
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength,suffix="Number of lines translated: {variables.nlines}",
                                    variables={'nlines':'-'})
    with open(monolingual_file,'r') as f:
        with open(pseudo_translation,'w') as w:
            for i,line in tqdm(enumerate(f)):
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


def get_translation_statistics(l1,l2):
    dictionary = pickle.load(open(os.path.join('Bilingual_Mappings',l1+'_'+l2+'_dict.pkl'),'rb'))
    dict_keys = list(dictionary.keys())
    dict_keys_used = set()
    total_words = 0
    words_translated = 0
    monolingual_file = l1+'_sentences.txt'
    pseudo_translation = l2+'_pseudo_'+l1+'.txt'
    bar = progressbar.ProgressBar(term_width = 20,suffix="Total Dict: {variables.total_dict} Used Dict: {variables.dict_used} Used%: {variables.used_perc} Total: {variables.total_words} Translated: {variables.translated}",
                                                                        variables = {'total_dict':'--','dict_used':'--','used_perc':'--','total_words':'--','translated':'--'})
    i = 0
    with open(monolingual_file,'r') as f1:
        with open(pseudo_translation,'r') as f2:
            line_l1 = next(f1)
            line_l2 = next(f2)
            while(line_l1 and line_l2):
                words_l1 = line_l1.split(" ")
                words_l2 = line_l2.split(" ")
                assert(len(words_l1)==len(words_l2))
                for (word_l1,word_l2) in zip(words_l1,words_l2):
                    total_words += 1
                    if word_l1 != word_l2:
                        words_translated+=1
                        if word_l1.endswith('\n'):
                            word_l1 = word_l1[:-1]
                        punctuations = ['...','।']+list(string.punctuation)
                        for punctuation in punctuations:
                            if word_l1.endswith(punctuation):
                                word_l1 = word_l1[:-1*len(punctuation)]
                                break
                        if word_l1 in dict_keys:
                            dict_keys_used.add(word_l1)
                # print(len(dict_keys_used))
                i+=1 
                bar.update(i+1,total_dict=str(len(dict_keys)),dict_used=str(len(dict_keys_used)),used_perc=str(len(dict_keys_used)/len(dict_keys)),
                                total_words=str(total_words),translated=str(words_translated))
                line_l1 = next(f1)
                line_l2 = next(f2)


if __name__=="__main__":
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    # dir_ = sys.argv[3]
    create_pseudo_translation(l1,l2)