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
from models.cnn import CNN
from utils.config import CFG
from utils.clean_text import *

from models.cnn import CNN
from models.classifier_onlyimage import Classifier_Emotion
#-----

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

def inference_emotion():
    # only get image
    test_data = MemoDataset_Emotion(test_images, None, humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest, \
        CFG.test_path, None, None, transform=None, task='emotion') 
    testloader = DataLoader(test_data, batch_size=CFG.batch_size, drop_last=False, shuffle=False, num_workers=4)

    model_path = '/home/tinvn/TIN/MEME_Challenge/code/temp_best/best_onlyimage/onlyimage_fold0_emotion_best_epoch25_7033.pth'
    states = torch.load(model_path,  map_location=torch.device('cpu'))
    model = CNN(is_pretrained=True, type_=CFG.cnn_type)# B, C, W,H B, 1792, 7, 7
    classifier = Classifier_Emotion(hidden_d=1792, hidden_d2=500, n_classes=CFG.n_emotion_classes, dropout=CFG.dropout, use_cbam=True)
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
        for i_batch, batch_dict in enumerate(testloader):
            model.eval()
            classifier.eval()
            # Forward pass
            images = batch_dict['x_images'].to(CFG.device)
            labels_humour = batch_dict['y_humour'].to(CFG.device)
            labels_sarcasm = batch_dict['y_sarcasm'].to(CFG.device)
            labels_offensive = batch_dict['y_offensive'].to(CFG.device)
            labels_motivation = batch_dict['y_motivation'].to(CFG.device)
            
            features = model(images)
            outputs_humour, outputs_sarcasm, outputs_offensive, outputs_motivation = classifier(features)
            
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

# inference
inference_emotion()

