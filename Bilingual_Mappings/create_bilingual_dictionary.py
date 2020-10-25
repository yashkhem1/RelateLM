import os
import sys
import random
import pickle

def get_bilingual_dictionary(l1, l2,choose_random=True):
    dir1 = l1+'-'+l2
    dir2 = l2+'-'+l1
    l1_l2_dict = {}
    l2_l1_dict = {}
    
    if os.path.exists(dir1):
        for file_ in os.listdir(dir1):
            with open(os.path.join(dir1,file_),encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                l1_word = line.split(":")[0].strip()
                l2_words = line.split(":")[1][1:].split(" ")
                if choose_random:
                    l2_word = l2_words[0].strip()
                
                l1_l2_dict[l1_word] = l2_word
                
                for word in l2_words:
                    word = word.strip()
                    l2_l1_dict[word] = l1_word

    if os.path.exists(dir2):
        for file_ in os.listdir(dir2):
            with open(os.path.join(dir2,file_),encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                l2_word = line.split(":")[0].strip()
                l1_words = line.split(":")[1][1:].split(" ")
                if choose_random:
                    l1_word = l1_words[0].strip()
                
                l2_l1_dict[l2_word] = l1_word

                for word in l1_words:
                    if word not in l1_l2_dict:
                        word = word.strip()
                        l1_l2_dict[word] = l2_word
    
    print("Size of",l1,"-",l2,"dictionary:",len(l1_l2_dict.keys()))
    print("Size of",l2,"-",l1,"dictionary:",len(l2_l1_dict.keys()))

    with open(l1+"_"+l2+"_dict.pkl",'wb') as w:
        pickle.dump(l1_l2_dict,w)

    with open(l2+"_"+l1+"_dict.pkl",'wb') as w:
        pickle.dump(l2_l1_dict,w)
    

if __name__ == "__main__":
    l1 = sys.argv[1]
    l2 = sys.argv[2]
    get_bilingual_dictionary(l1,l2)
