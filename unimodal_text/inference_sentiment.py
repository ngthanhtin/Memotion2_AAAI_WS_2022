
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
from utils.dataset_unimodal import MemoDataset_Sentiment
from utils.config import CFG
from utils.clean_text import *

from transformers import BertTokenizer, RobertaTokenizer,RobertaModel, XLNetTokenizer, RobertaTokenizer, \
    BertForSequenceClassification, XLNetForSequenceClassification, RobertaModel, AdamW, RobertaForSequenceClassification

# manually fix batch size
CFG.batch_size = 10
CFG.test_path = CFG.private_test_path
CFG.test_csv_path = CFG.private_csv_test_path

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

def inference_sentiment():

    test_data = MemoDataset_Sentiment(None, xtest, target_tensor_test, CFG.test_path, roberta_tokenizer, CFG.max_len, transform=get_transforms(data = 'train')) 
    
    testloader = DataLoader(test_data, batch_size=CFG.batch_size, drop_last=False, shuffle=False, num_workers=4)
    #load full model
    # onlytext_fold0_sentiment_best.pth
    model_path = '/home/tinvn/TIN/MEME_Challenge/code/temp_best/best_onlytext/onlytext_fold0_sentiment_best_5154_new.pth'
    states = torch.load(model_path, map_location = torch.device('cpu'))

    model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=CFG.n_sentiment_classes)
    model.to(CFG.device)
    #load model
    model.load_state_dict(states['model'])
    model.to(CFG.device)
    
    best_acc = 0.
    # Validate the model
        
    target_total_test = []
    predicted_total_test = [] 
    ids_test = []
    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, batch_dict in enumerate(testloader):
            model.eval()
            indices = batch_dict['x_indices'].to(CFG.device)
            attn_mask = batch_dict['x_attn_mask'].to(CFG.device)
            labels = batch_dict['y_target'].to(CFG.device)

            outputs = model(input_ids=indices, token_type_ids=None, attention_mask=attn_mask)
            logits = outputs[0]
            # max returns (value ,index)
            _, predicted = torch.max(logits.data, 1)
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
roberta_tokenizer = tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

###----- LOAD DATA---------###
testdata = pd.read_csv(CFG.test_csv_path, header=None) 

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

# inference
inference_sentiment()

