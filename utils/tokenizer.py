import sys
sys.path.insert(0, './')

import re
import pandas as pd
import numpy as np
import torch

from utils.clean_text import scrub_words
from utils.config import CFG

#### Tokenizer ####
class Tokenizer():
    def __init__(self, sentences):
        self.sentences = sentences
        self.stoi = {}
        self.itos = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for s in self.sentences:
            # update with individual tokens
            self.vocab.update(str(s).split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.stoi['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.stoi[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.stoi.items():
            self.itos[index] = word  


def create_tokenizer():
    ### Construct tokenizer ###
    ###--- LOAD DATA-----------###
    traindata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_train.csv', header=None)
    testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 

    x = traindata[:][2].to_numpy()[1:] # text
    
    xtest = testdata[:][2].to_numpy()[1:] # text

    #-------SCRUB WORDS-----------------
    x = [element.lower() if isinstance(element, str) else element for element in x]
    xtest = [element.lower() if isinstance(element, str) else element for element in xtest]

    x = [scrub_words(element) if isinstance(element, str) else element for element in x]
    xtest = [scrub_words(element) if isinstance(element, str) else element for element in xtest]
    print("Finish scrubing words...")

    # construct vocab and indexing
    inputs = Tokenizer(np.concatenate((x, xtest), axis=0))

    torch.save(inputs, 'tokenizer2.pth')
    print('Saved tokenizer')

if __name__ == "__main__":
    create_tokenizer()