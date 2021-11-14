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
from utils.dataset_unimodal import MemoDataset_Emotion
from utils.tokenizer import Tokenizer
# import models
from models.model_cnnbert import CNN_Roberta_Concat, CNN_Roberta_SAN
from transformers import BertTokenizer, RobertaTokenizer
###
from utils.config import CFG
from utils.clean_text import *
# manually fix batch size
CFG.batch_size = 10
CFG.model_name = 'cnnbert_san'
CFG.device = 'cuda:1'

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

def inference_emotion():
 
    test_data = MemoDataset_Emotion(np.array(test_images), xtest, humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest, \
        CFG.test_path, roberta_tokenizer, CFG.max_len, transform=None, task="emotion") 
    
    testloader = DataLoader(test_data, batch_size=CFG.batch_size, drop_last=False, shuffle=False, num_workers=4)


    ### Model
    if CFG.model_name == 'cnnbert_concat':
        model = CNN_Roberta_Concat(roberta_model_name = 'distilroberta-base', cnn_type = CFG.cnn_type, num_classes=4)
    elif CFG.model_name == 'cnnbert_san':
        model = CNN_Roberta_SAN(roberta_model_name = 'distilroberta-base', cnn_type = CFG.cnn_type)
    
    #load full model
    states = torch.load(f'{CFG.model_name}_fold0_emotion_best.pth', map_location = torch.device('cpu'))
    model.load_state_dict(states['model'])
    model.to(CFG.device)
        
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
        for i_batch, batch_dict in enumerate(testloader):
            model.eval()
            
            # Forward pass
            images = batch_dict['x_images'].to(CFG.device)
            indices = batch_dict['x_indices'].to(CFG.device)
            attn_mask = batch_dict['x_attn_mask'].to(CFG.device)
            labels_humour = batch_dict['y_humour'].to(CFG.device)
            labels_sarcasm = batch_dict['y_sarcasm'].to(CFG.device)
            labels_offensive = batch_dict['y_offensive'].to(CFG.device)
            labels_motivation = batch_dict['y_motivation'].to(CFG.device)
            
           
            outputs = model(indices, attn_mask, images)

            labels = [] # aggregated labels such as: [[0,0, 1, 1], [0,1, 0, 1], , [1,1, 1, 1], [1,0, 1,0], [1,0, 1, 1]]
            for k in range(len(labels_humour)):
                labels.append([labels_humour[k], labels_sarcasm[k], labels_offensive[k], labels_motivation[k]])
            labels = torch.Tensor(labels).to(CFG.device)
            
            pred_label = torch.sigmoid(outputs).detach().cpu().numpy()
            # Calculate Accuracy
            threshold = 0.50
            pred_bools = np.where(pred_label > threshold, 1., 0.)
            # max returns (value ,index)
            predicted_humour = torch.from_numpy(pred_bools[:,0]).to(CFG.device)
            predicted_sarcasm = torch.from_numpy(pred_bools[:,1]).to(CFG.device)
            predicted_offensive = torch.from_numpy(pred_bools[:,2]).to(CFG.device)
            predicted_motivation = torch.from_numpy(pred_bools[:,3]).to(CFG.device)
            
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
            f.write("{}{}{}{}\n".format(int(h),int(s),int(o),int(m)))
        f.close()

### --- LOAD TOKENIZER----- ###
roberta_tokenizer = tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=True)

###--- LOAD DATA-----------###
testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 

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

# inference
inference_emotion()



        
        


    





