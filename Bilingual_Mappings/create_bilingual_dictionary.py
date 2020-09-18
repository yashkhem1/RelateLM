import os
import sys
import random
import pickle

def get_bilingual_dictionary(l1, l2,choose_random=True):
    dir1 = l1+'-'+l2
    dir2 = l2+'-'+l1
    l1_l2_dict = {}
    l2_l1_dict = {}
    for file_ in os.listdir(dir1):
        with open(os.path.join(dir1,file_),encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            l1_word = line.split(":")[0]
            l2_words = line.split(":")[1][1:].split(" ")
            if choose_random:
                l2_word = l2_words[0]
                if l2_word.endswith('\n'):
                    l2_word = str(l2_word)[:-1]
            
            if l1_word not in l1_l2_dict.keys():
                l1_l2_dict[l1_word] = l2_word
            
            if l2_word not in l2_l1_dict.keys():
                l2_l1_dict[l2_word] = l1_word

    for file_ in os.listdir(dir2):
        with open(os.path.join(dir2,file_),encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            l2_word = line.split(":")[0]
            l1_words = line.split(":")[1][1:].split(" ")
            if choose_random:
                l1_word = l1_words[0]
                if l1_word.endswith('\n'):
                    l1_word = str(l1_word)[:-1]
            
            if l1_word not in l1_l2_dict.keys():
                l1_l2_dict[l1_word] = l2_word
            
            if l2_word not in l2_l1_dict.keys():
                l2_l1_dict[l2_word] = l1_word

    with open(l1+"_"+l2+"_dict.pkl",'wb') as w:
        pickle.dump(l1_l2_dict,w)

    with open(l2+"_"+l1+"_dict.pkl",'wb') as w:
        pickle.dump(l2_l1_dict,w)
    

if __name__ == "__main__":
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    get_bilingual_dictionary(l1,l2)
