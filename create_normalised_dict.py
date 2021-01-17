from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
from itertools import groupby
import argparse
import pickle

def normaliser(input_str):
  normalizer = normalizers.Sequence([NFD(), StripAccents()])
  output_str=normalizer.normalize_str(input_str)
  output_str=output_str.replace('ष','श').replace('ळ','ल').replace('ण','न').replace('ङ','').replace('ञ','').replace('_','')
  output_str=''.join([i[0] for i in groupby(output_str)])
  return output_str

def dict_normaliser(input_dict_path, normalised_output_dict_path):
  input_dict = pickle.load(open(input_dict_path,'rb'))
  normalised_output_dict={}
  for key in input_dict:
    if normaliser(key) not in normalised_output_dict:
      normalised_output_dict[normaliser(key)] = [[],[]]
    new_indices = [x for x in range(len(input_dict[key][0])) if input_dict[key][0][x] not in normalised_output_dict[normaliser(key)]]
    normalised_output_dict[normaliser(key)][0] += [input_dict[key][0][i] for i in new_indices]
    normalised_output_dict[normaliser(key)][1] += [input_dict[key][1][i] for i in new_indices]
  with open(normalised_output_dict_path,'wb') as w:
    pickle.dump(normalised_output_dict,w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--o',type=str,help='Output path to normalised dict pkl file')
    parser.add_argument('--i',type=str,help='Path to input dict pkl file')
    args = parser.parse_args()
    dict_normaliser(args.i,args.o)