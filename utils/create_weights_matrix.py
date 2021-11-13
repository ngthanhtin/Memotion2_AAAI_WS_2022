import sys
sys.path.insert(0, './')
import torch
from config import CFG
import bcolz, pickle
import numpy as np
from utils.tokenizer import Tokenizer

tokenizer = torch.load(CFG.tokenizer_path)

vectors = bcolz.open(f'./glove/glove.wiki200d.dat')[:] 
words = pickle.load(open(f'./glove/glove.wiki_words_200d.pkl', 'rb')) 
word2idx = pickle.load(open(f'./glove/glove.wiki_idx_200d.pkl', 'rb'))
 
glove = {w: vectors[word2idx[w]] for w in words}

#####
# Here I use the Glove Embedding for the vocabulary of this training and testing set
##### 

matrix_len = len(tokenizer.stoi)
weights_matrix = np.zeros((matrix_len, 200))
words_found = 0

for i, word in enumerate(tokenizer.vocab):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(CFG.embedding_dim, ))
    
with open('weights_matrix.npy', 'wb') as f:
    np.save(f, weights_matrix)