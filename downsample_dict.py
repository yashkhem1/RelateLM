import pickle
import random
import argparse

def downsample_dict(dictpath,n,outpath):
    input_dict = pickle.load(open(dictpath,'rb'))
    keys = list(input_dict.keys())
    keys_downsampled = random.sample(keys,n)
    output_dict = {}
    for key in keys_downsampled:
        output_dict[key] = input_dict[key]
    with open(outpath,'wb') as w:
        pickle.dump(output_dict,w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dictpath',type=str,help='Input dict path')
    parser.add_argument('--n',type=int,help='Number of keys')
    parser.add_argument('--outpath',type=str,help='Path to output dictionary file')
    args = parser.parse_args()
    downsample_dict(args.dictpath,args.n,args.outpath)