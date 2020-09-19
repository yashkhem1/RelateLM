import torch
from torch.utils.data import IterableDataset,DataLoader, Dataset
from utils import get_token_mappings_line
import os
import sys

class PseudoTokenMappingDataset(IterableDataset):
    def __init__(self,file1,file2):
        self.file1 = file1
        self.file2 = file2

    def preprocess(self,l1, l2):
        enc1, enc2, tok_map = get_token_mappings_line(l1,l2)
        return zip(enc1,enc2,tok_map)

    def timepass(self,l1):
        return l1

    def __iter__(self):
            l1_iter = open(self.file1)
            l2_iter = open(self.file2)
            token_map_iter = map(self.timepass,l1_iter)

            return token_map_iter

class CustomDataset(Dataset):
    # A pytorch dataset class for holding data for a text classification task.
    def __init__(self, filename):
        '''
        Takes as input the name of a file containing sentences with a classification label (comma separated) in each line.
        Stores the text data in a member variable X and labels in y
        '''

        #Opening the file and storing its contents in a list
        with open(filename) as f:
            lines = f.read().split('\n')

        #Splitting the text data and lables from each other
        X, y = [], []
        for line in lines:
            X.append(line.split(',')[0])
            y.append(line.split(',')[0])
            
        #Store them in member variables.
        self.X = X
        self.y = y

    def preprocess(self, text):

        ### Do something with text here
        text_pp = text.lower().strip()
        ###

        return text_pp
    
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self, index):

       '''
       Returns the text and labels present at the specified index of the lists.
       '''

       return self.preprocess(self.X[index]), self.y[index]


    
if __name__=="__main__":
    # dataset = PseudoTokenMappingDataset('Hindi_sentences.txt','Marathi_pseudo_Hindi.txt')
    # dl = DataLoader(dataset=dataset,batch_size=5)
    # for i,(enc1) in enumerate(dl):
    #     print(enc1)
    #     if i==0:
    #         break
    dataset = CustomDataset('original_hindi.txt')

    #Wrap it around a dataloader
    dataloader = DataLoader(dataset, batch_size = 2, num_workers = 1)
    for i,(x,y) in enumerate(dataloader):
        print(x)