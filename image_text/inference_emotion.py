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

import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.transformation import get_transforms
from utils.dataset import MemoDataset_Emotion
from utils.tokenizer import Tokenizer
# import models
from models.model_mha import MemoLSTM_MHA
from models.model_san import MemoLSTM_SAN
from models.model_cnnbert import CNN_Roberta_Concat, CNN_Roberta_SAN
###
from models.classifier import ClassifierLSTM_Emotion
from utils.config import CFG
from utils.clean_text import *

# manually fix batch size
CFG.batch_size = 10
CFG.model_name = 'multihop'
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

def inference_emotion():
    test_data = MemoDataset_Emotion(test_images, input_tensor_test_pad, humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest, \
                                                    root_dir = CFG.test_path, transform=None, task='emotion')   
    testloader = DataLoader(test_data, batch_size=CFG.batch_size, drop_last=False, shuffle=False, num_workers=4)

    #load full model
    model_path = '/home/tinvn/TIN/MEME_Challenge/code/temp_best/best_image_text/multihop/pretrained true/multihop_fold0_emotion_best_7107_epoch20.pth'
    # model_path = '/home/tinvn/TIN/MEME_Challenge/code/temp_best/best_image_text/san/pretrained true/san_fold0_emotion_best_714_epoch8.pth'
    states = torch.load(model_path, map_location = torch.device('cpu'))
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
        
    classifier = ClassifierLSTM_Emotion(CFG.hidden_d, CFG.hidden_d2, CFG.n_emotion_classes, CFG.dropout, CFG.n_layers)
    #load model and classifier
    model.load_state_dict(states['model'])
    classifier.load_state_dict(states['classifier'])
    model.to(CFG.device)
    classifier.to(CFG.device)

    target_total_test_humour = []
    predicted_total_test_humour = []
    
    target_total_test_sarcasm = []
    predicted_total_test_sarcasm = []
    
    target_total_test_offensive = []
    predicted_total_test_offensive = []
    
    target_total_test_motivation = []
    predicted_total_test_motivation = [] 

    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i_batch, (images, input_ids, labels_humour, labels_sarcasm, labels_offensive, labels_motivation) in enumerate(testloader):
            model.eval()
            classifier.eval()
            images = images.to(CFG.device)
            labels_humour = labels_humour.to(CFG.device)
            labels_sarcasm = labels_sarcasm.to(CFG.device)
            labels_offensive = labels_offensive.to(CFG.device)
            labels_motivation = labels_motivation.to(CFG.device)

            mm_atmf_outputs = []

            input_utt = torch.tensor(input_ids[0]).to(CFG.device)
            atmf_output = model(images, input_utt)
            mm_atmf_outputs.append(atmf_output)

            lstm_input = torch.stack(mm_atmf_outputs, dim=0)
            outputs_humour, outputs_sarcasm, outputs_offensive, outputs_motivation = classifier(lstm_input)
            
            _, predicted_humour = torch.max(outputs_humour.data, 1)
            _, predicted_sarcasm = torch.max(outputs_sarcasm.data, 1)
            _, predicted_offensive = torch.max(outputs_offensive.data, 1)
            _, predicted_motivation = torch.max(outputs_motivation.data, 1)
            
            target_total_test_humour.append(labels_humour)
            predicted_total_test_humour.append(predicted_humour)
        
            target_total_test_sarcasm.append(labels_sarcasm)
            predicted_total_test_sarcasm.append(predicted_sarcasm)
        
            target_total_test_offensive.append(labels_offensive)
            predicted_total_test_offensive.append(predicted_offensive)
        
            target_total_test_motivation.append(labels_motivation)
            predicted_total_test_motivation.append(predicted_motivation)
            
            

            target_inter_humour = [t.cpu().numpy() for t in target_total_test_humour]
            predicted_inter_humour = [t.cpu().numpy() for t in predicted_total_test_humour]
            target_inter_humour =  np.stack(target_inter_humour, axis=0).ravel()
            predicted_inter_humour =  np.stack(predicted_inter_humour, axis=0).ravel()
        
            target_inter_sarcasm = [t.cpu().numpy() for t in target_total_test_sarcasm]
            predicted_inter_sarcasm = [t.cpu().numpy() for t in predicted_total_test_sarcasm]
            target_inter_sarcasm =  np.stack(target_inter_sarcasm, axis=0).ravel()
            predicted_inter_sarcasm =  np.stack(predicted_inter_sarcasm, axis=0).ravel()
        
            target_inter_offensive = [t.cpu().numpy() for t in target_total_test_offensive]
            predicted_inter_offensive = [t.cpu().numpy() for t in predicted_total_test_offensive]
            target_inter_offensive =  np.stack(target_inter_offensive, axis=0).ravel()
            predicted_inter_offensive =  np.stack(predicted_inter_offensive, axis=0).ravel()
        
            target_inter_motivation = [t.cpu().numpy() for t in target_total_test_motivation]
            predicted_inter_motivation = [t.cpu().numpy() for t in predicted_total_test_motivation]
            target_inter_motivation =  np.stack(target_inter_motivation, axis=0).ravel()
            predicted_inter_motivation =  np.stack(predicted_inter_motivation, axis=0).ravel()

            
        current_macro_humour = f1_score(target_inter_humour, predicted_inter_humour, average="macro")
        current_macro_sarcasm = f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro")
        current_macro_offensive = f1_score(target_inter_offensive, predicted_inter_offensive, average="macro")
        current_macro_motivation = f1_score(target_inter_motivation, predicted_inter_motivation, average="macro")
        
        current_micro_humour = f1_score(target_inter_humour, predicted_inter_humour, average="micro")
        current_micro_sarcasm = f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro")
        current_micro_offensive = f1_score(target_inter_offensive, predicted_inter_offensive, average="micro")
        current_micro_motivation = f1_score(target_inter_motivation, predicted_inter_motivation, average="micro")
        
        current_macro_average = (current_macro_humour + current_macro_sarcasm + current_macro_offensive + current_macro_motivation)/4 

        print(f' Humour Micro F1 on test set: {current_micro_humour}')
        print(f' Humour Macro F1 on test set: {current_macro_humour}')
        print(f' Sarcasm Micro F1 on test set: {current_micro_sarcasm}')
        print(f' Sarcasm Macro F1 on test set: {current_macro_sarcasm}')
        print(f' Offensive Micro F1 on test set: {current_micro_offensive}')
        print(f' Offensive Macro F1 on test set: {current_macro_offensive}')
        print(f' Motivation Micro F1 on test set: {current_micro_motivation}')
        print(f' Motivation Macro F1 on test set: {current_macro_motivation}')
        print(f' Current Average Macro F1 on test set: {current_macro_average}')
        
        print(confusion_matrix(target_inter_humour, predicted_inter_humour))
        print(confusion_matrix(target_inter_sarcasm, predicted_inter_sarcasm))
        print(confusion_matrix(target_inter_offensive, predicted_inter_offensive))
        print(confusion_matrix(target_inter_motivation, predicted_inter_motivation))

        f = open("./results/emotion_result.txt", "w")
        for h,s,o,m in zip(predicted_inter_humour, predicted_inter_sarcasm, predicted_inter_offensive, predicted_inter_motivation):
            f.write("{}{}{}{}\n".format(h,s,o,m))
        f.close()

### --- LOAD TOKENIZER----- ###
tokenizer = torch.load(CFG.tokenizer_path)

###--- LOAD DATA-----------###
testdata = pd.read_csv(CFG.test_csv_path, header=None) 
#--test data---
test_images = testdata[:][0].to_numpy()[1:] # image paths      
xtest = testdata[:][2].to_numpy()[1:] # text
humour_ytest = testdata[:][3].to_numpy()[1:]
sarcasm_ytest = testdata[:][4].to_numpy()[1:]
offensive_ytest = testdata[:][5].to_numpy()[1:]
motivational_ytest = testdata[:][6].to_numpy()[1:]

def textlabel_to_digitallabel(humour_arr, sarcasm_arr, offensive_arr, motivational_arr):
    for idx in range(len(humour_arr)):
        if (humour_arr[idx] == 'not_funny'):
            humour_arr[idx] = 0
        if (humour_arr[idx] == 'funny' or humour_arr[idx] =='very_funny' or humour_arr[idx] =='hilarious'):
            humour_arr[idx] = 1  

    for idx in range(len(sarcasm_arr)):
        if (sarcasm_arr[idx] == 'not_sarcastic'):
            sarcasm_arr[idx] = 0
        if (sarcasm_arr[idx] == 'very_sarcastic' or sarcasm_arr[idx] =='little_sarcastic' or sarcasm_arr[idx] =='extremely_sarcastic'):
            sarcasm_arr[idx] = 1    

    for idx in range(len(offensive_arr)):
        if (offensive_arr[idx] == 'not_offensive'):
            offensive_arr[idx] = 0
        if (offensive_arr[idx] == 'slight' or offensive_arr[idx] =='hateful_offensive' or offensive_arr[idx] =='very_offensive'):
            offensive_arr[idx] = 1    
            
    for idx in range(len(motivational_arr)):
        if (motivational_arr[idx] == 'not_motivational'):
            motivational_arr[idx] = 0
        if (motivational_arr[idx] == 'motivational'):
            motivational_arr[idx] = 1 

    return humour_arr, sarcasm_arr, offensive_arr, motivational_arr

humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest = textlabel_to_digitallabel(humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest)

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
inference_emotion()



        
        


    





