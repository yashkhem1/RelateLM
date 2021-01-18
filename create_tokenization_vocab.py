import pickle
import argparse

def create_tokenization_vocab(dict_path, original_vocab_path, output_vocab_path):
    dictionary = pickle.load(open(dict_path,'rb'))
    with open(original_vocab_path,'r') as f:
        lines = f.readlines()
    tokenization_vocab = [x.strip() for x in lines if not x.strip().startswith('[unused')]
    tokenization_vocab += [x for x in dictionary if x not in tokenization_vocab]

    with open(output_vocab_path,'w') as w:
        for vocab in tokenization_vocab:
            w.write(vocab+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_path',type=str,help='Path to dictionary file')
    parser.add_argument('--orig_vocab_path',type=str,help='Path to input vocab file')
    parser.add_argument('--out_vocab_path',type=str,help='Path to output vocab file')
    args = parser.parse_args()
    create_tokenization_vocab(args.dict_path,args.orig_vocab_path,args.out_vocab_path)