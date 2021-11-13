
import sys
sys.path.insert(0, './')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random, os
import nltk
import xlrd
# nltk.download('punkt')

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.transformation import get_transforms
from utils.dataset import MemoDataset_Sentiment
from utils.tokenizer import Tokenizer
# import models
from models.model_mha import MemoLSTM_MHA
from models.model_san import MemoLSTM_SAN
from models.model_cnnbert import CNN_Roberta_Concat, CNN_Roberta_SAN
###
from models.classifier import ClassifierLSTM_Sentiment
from utils.config import CFG
from utils.clean_text import *

# manually fix batch size
CFG.batch_size = 10
CFG.model_name = 'san'
print("Inference {}".format(CFG.model_name))

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

def inference_sentiment():

    test_data = MemoDataset_Sentiment(test_images, input_tensor_test_pad, target_tensor_test, root_dir = CFG.test_path, transform=None) 
    
    testloader = DataLoader(test_data, batch_size=CFG.batch_size, drop_last=False, shuffle=False, num_workers=4)
    #load full model
    states = torch.load(f'{CFG.model_name}_fold0test_sentiment_best.pth', map_location = torch.device('cpu'))

    if CFG.model_name == 'multihop':
        model = MemoLSTM_MHA(CFG.batch_size, CFG.n_sentiment_classes, CFG.units, len(tokenizer.stoi), CFG.embedding_dim, CFG.hidden_d, \
         CFG.dropout, CFG.n_layers, CFG.cnn_type, CFG.device)
    elif CFG.model_name == 'san':
        model = MemoLSTM_SAN(CFG.batch_size, CFG.n_sentiment_classes, CFG.units, len(tokenizer.stoi), CFG.embedding_dim, CFG.hidden_d, \
         CFG.dropout, CFG.n_layers, CFG.cnn_type, CFG.device)
    elif CFG.model_name == 'cnnbert_concat':
        model = CNN_Roberta_Concat(roberta_model_name = 'distilroberta-base', cnn_type = CFG.cnn_type, num_classes=CFG.n_sentiment_classes)
    elif CFG.model_name == 'cnnbert_san':
        model = CNN_Roberta_SAN(roberta_model_name = 'distilroberta-base', cnn_type = CFG.cnn_type, num_classes=CFG.n_sentiment_classes)

    classifier = ClassifierLSTM_Sentiment(CFG.hidden_d, CFG.hidden_d2, CFG.dropout, CFG.n_layers, CFG.n_sentiment_classes, CFG.device)
    #load model and classifier
    model.load_state_dict(states['model'])
    classifier.load_state_dict(states['classifier'])
    model.to(CFG.device)
    classifier.to(CFG.device)

    best_acc = 0.
    # Validate the model
        
    target_total_test = []
    predicted_total_test = [] 
    ids_test = []
    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i_batch, (images, input_ids, labels) in enumerate(testloader):
            model.eval()
            classifier.eval()
            images = images.to(CFG.device)
            labels = labels.to(CFG.device)

            mm_atmf_outputs = []

            input_utt = torch.tensor(input_ids[0]).to(CFG.device)
            atmf_output = model(images, input_utt)
            mm_atmf_outputs.append(atmf_output)

            lstm_input = torch.stack(mm_atmf_outputs, dim=0)
            outputs = classifier(lstm_input)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            target_total_test.append(labels)
            predicted_total_test.append(predicted)
            
            target_inter = [t.cpu().numpy() for t in target_total_test]
            predicted_inter = [t.cpu().numpy() for t in predicted_total_test]
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            
        current_macro = f1_score(target_inter, predicted_inter, average="macro")
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the test set is: {acc} %')        
        print(f' Micro F1 on the testing: {f1_score(target_inter, predicted_inter, average="micro")}')
        print(f' Macro F1 on the testing: {f1_score(target_inter, predicted_inter, average="macro")}')
        print(confusion_matrix(target_inter, predicted_inter))   
        
        f = open("./results/sentiment_result.txt", "w")
        for pred in predicted_inter:
            f.write("{}\n".format(pred))
        f.close()

### --- LOAD TOKENIZER----- ###
tokenizer = torch.load(CFG.tokenizer_path)

###----- LOAD DATA---------###
testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 

test_images = testdata[:][0].to_numpy()[1:]
xtest = testdata[:][1].to_numpy()[1:]
ytest = testdata[:][7].to_numpy()[1:]

#------ENCODER LABEL---------
labelencode = LabelEncoder()

ylabeltest = labelencode.fit_transform(ytest.astype(str))
target_tensor_test = (ylabeltest.tolist())

#-------SCRUB WORDS-----------------
xtest = [element.lower() if isinstance(element, str) else element for element in xtest]
xtest = [scrub_words(element) if isinstance(element, str) else element for element in xtest]
print("Finish scrubing words...")

#---------VECTORIZE TENSOR-----------

input_tensor_test = []
for idx in range(len(xtest)):
    input_tensor_testsample = [tokenizer.stoi.get(s) for s in str(xtest[idx]).split(' ')]
    input_tensor_test.append(input_tensor_testsample)
print("Finish vectorizing tensor...")

#### REPRESENT OUT-OF-VOCAB WORD BY A SPACE ####

for idx in range(len(input_tensor_test)):
    for v in range(len(input_tensor_test[idx])):
        if (input_tensor_test[idx][v] == None):
            input_tensor_test[idx][v] = 1
print("Finish replace out-of-vocabulary...")

#### PADDING TO MAX_LEN ####    
def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded

# inplace padding
input_tensor_test_pad = []

for idx in range(len(input_tensor_test)):
    xsampletest = [pad_sequences(input_tensor_test[idx], CFG.max_len)]
    input_tensor_test_pad.append(xsampletest)
print("Finish padding...")

# inference
inference_sentiment()

