from nltk.translate.bleu_score import sentence_bleu
import string
from nltk.translate.bleu_score import SmoothingFunction
import argparse

def avg_bleu(mono_path, pseudo_path):
    with open(mono_path,'r') as f:
        lines1 = f.readlines()
    with open(pseudo_path,'r') as f:
        lines2 = f.readlines()
    assert(len(lines1)==len(lines2))
    sum=0
    valid_lines=0
    for x in range(len(lines1)):
        s1=''.join(i for i in lines1[x] if not(i in string.punctuation+'\n'))
        s2=''.join(i for i in lines2[x] if not(i in string.punctuation+'\n'))
        if not (not s1 or not s2):
            valid_lines=valid_lines+1
            if(s1==s2):
                bleu=1
            else:
                bleu=sentence_bleu([s1.split()], s2.split(),smoothing_function=SmoothingFunction().method4)
            sum=sum+bleu
    avg_bleu=sum/valid_lines
    print(avg_bleu)
    return avg_bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mono',type=str,help='Input path to monoingual data txt file')
    parser.add_argument('--pseudo',type=str,help='Input Path to pseudo txt file')
    args = parser.parse_args()
    avg_bleu(args.mono, args.pseudo)






# with open("/mnt/Multilingual-Models-for-Related-Languages/Wikipedia/hin_pseudo_ben_trans_hin_wik_weighted.txt",'r') as f:
#         lines2 = f.readlines()



# with open("/mnt/Multilingual-Models-for-Related-Languages/Wikipedia/Bengali_trans_Hindi.txt",'r') as f:
#         lines1 = f.readlines()

# with open("/mnt/Multilingual-Models-for-Related-Languages/Wikipedia/hin_pseudo_ben_trans_hin_wik_weighted.txt",'r') as f:
#         lines2 = f.readlines()

# s=0
# i=0
# assert(len(lines1)==len(lines2))

# for x in range(len(lines1)):
#     s1=''.join(i for i in lines1[x] if not(i in string.punctuation+'\n'))
#     s2=''.join(i for i in lines2[x] if not(i in string.punctuation+'\n'))
#     if not (not s1 or not s2):
#         i=i+1
#         if(s1==s2):
#             bleu=1
#         else:
#             bleu=sentence_bleu([s1.split()], s2.split(),smoothing_function=SmoothingFunction().method4)
#         s=s+bleu

# print(s)
# print(i)
# print(s/i)




