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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torch
import torch.nn as nn

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

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
CFG.batch_size = 8

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

def train_loop(trn_idx, val_idx):

    print('Training with fold {} started'.format(train_fold))
    #train
    train_humour = [humour_y[index] for index in trn_idx]
    train_sarcasm = [sarcasm_y[index] for index in trn_idx]
    train_offensive = [offensive_y[index] for index in trn_idx]
    train_motivational = [motivational_y[index] for index in trn_idx]
    #val
    val_humour = [humour_y[index] for index in val_idx]
    val_sarcasm = [sarcasm_y[index] for index in val_idx]
    val_offensive = [offensive_y[index] for index in val_idx]
    val_motivational = [motivational_y[index] for index in val_idx]

    # only get image
    train_data = MemoDataset_Emotion(train_images[trn_idx], None, train_humour, train_sarcasm, train_offensive, train_motivational \
        , CFG.train_path, None, None, transform=get_transforms(data = 'train'), task='emotion') 
    test_data = MemoDataset_Emotion(train_images[val_idx], None, val_humour, val_sarcasm, val_offensive, val_motivational, \
        CFG.train_path, None, None, transform=None, task='emotion') 
    # 
    trainloader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True, drop_last = True, num_workers=4) # if have sampler, dont use shuffle
    testloader = DataLoader(test_data, batch_size=CFG.batch_size, drop_last=False, shuffle=False, num_workers=4)
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, 
                                          T_max      = CFG.T_max, 
                                          eta_min    = CFG.min_lr, 
                                          last_epoch = -1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                    T_0        = CFG.T_0, 
                                                    T_mult     = 1, 
                                                    eta_min    = CFG.min_lr, 
                                                    last_epoch = -1)
        return scheduler

    # states = torch.load(CFG.sentiment_model,  map_location=torch.device('cpu'))
    model = CNN(is_pretrained=False, type_=CFG.cnn_type)# B, C, W,H B, 1792, 7, 7
    classifier = Classifier_Emotion(hidden_d=1792, hidden_d2=500, n_classes=CFG.n_emotion_classes, dropout=CFG.dropout, use_cbam=True)
    # model.load_state_dict(states['model'])
    model.to(CFG.device)
    classifier.to(CFG.device)
    params = list(model.parameters()) + list(classifier.parameters())
    # Loss and optimizer
    criterion_humour = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_humour)
    criterion_sarcasm = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_sarcasm)
    criterion_offensive = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_offensive)
    criterion_motivation = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_motivation)

    optimizer = torch.optim.Adam(params, lr = CFG.learning_rate, weight_decay=CFG.weight_decay)
    # optimizer = torch.optim.AdamW(params, lr=CFG.learning_rate, betas=(0.9, 0.999))
    scheduler = get_scheduler(optimizer)

    batch_wise_loss = []
    batch_wise_micro_f1_humour = []
    batch_wise_macro_f1_humour = []
    epoch_wise_macro_f1_humour = []
    epoch_wise_micro_f1_humour = []

    batch_wise_micro_f1_sarcasm = []
    batch_wise_macro_f1_sarcasm = []
    epoch_wise_macro_f1_sarcasm = []
    epoch_wise_micro_f1_sarcasm = []

    batch_wise_micro_f1_offensive = []
    batch_wise_macro_f1_offensive = []
    epoch_wise_macro_f1_offensive = []
    epoch_wise_micro_f1_offensive = []

    batch_wise_micro_f1_motivation = []
    batch_wise_macro_f1_motivation = []
    epoch_wise_macro_f1_motivation = []
    epoch_wise_micro_f1_motivation = []

    epoch_wise_average_macro_f1 = []
    best_wise_average_macro_f1 = 0. # to save model

    # Train the model
    n_total_steps = len(trainloader)

    # Train the model
    n_total_steps = len(trainloader)
    for epoch in range(CFG.epochs):
        
        target_total_humour = []
        predicted_total_humour = []
        
        target_total_sarcasm = []
        predicted_total_sarcasm = []
        
        target_total_offensive = []
        predicted_total_offensive = []
        
        target_total_motivation = []
        predicted_total_motivation = []
        
        for i, batch_dict in enumerate(trainloader):  
            
            model.train()
            classifier.train()
            # Forward pass
            images = batch_dict['x_images'].to(CFG.device)
            labels_humour = batch_dict['y_humour'].to(CFG.device)
            labels_sarcasm = batch_dict['y_sarcasm'].to(CFG.device)
            labels_offensive = batch_dict['y_offensive'].to(CFG.device)
            labels_motivation = batch_dict['y_motivation'].to(CFG.device)
            
            features = model(images)
            outputs_humour, outputs_sarcasm, outputs_offensive, outputs_motivation = classifier(features)

            loss_humour = criterion_humour(outputs_humour, labels_humour)
            loss_sarcasm = criterion_sarcasm(outputs_sarcasm, labels_sarcasm)
            loss_offensive = criterion_offensive(outputs_offensive, labels_offensive)
            loss_motivation = criterion_motivation(outputs_motivation, labels_motivation)
            loss_average = (loss_humour + loss_sarcasm + loss_offensive + loss_motivation)/4
            
            # Backward and optimize
            optimizer.zero_grad()
            loss_average.backward()
            optimizer.step()
            
            # max returns (value ,index)
            _, predicted_humour = torch.max(outputs_humour.data, 1)
            _, predicted_sarcasm = torch.max(outputs_sarcasm.data, 1)
            _, predicted_offensive = torch.max(outputs_offensive.data, 1)
            _, predicted_motivation = torch.max(outputs_motivation.data, 1)
            
            target_total_humour.append(labels_humour)
            predicted_total_humour.append(predicted_humour)
            
            target_total_sarcasm.append(labels_sarcasm)
            predicted_total_sarcasm.append(predicted_sarcasm)
            
            target_total_offensive.append(labels_offensive)
            predicted_total_offensive.append(predicted_offensive)
            
            target_total_motivation.append(labels_motivation)
            predicted_total_motivation.append(predicted_motivation)
            
            
            
            target_inter_humour = [t.cpu().numpy() for t in target_total_humour]
            predicted_inter_humour = [t.cpu().numpy() for t in predicted_total_humour]
            target_inter_humour =  np.stack(target_inter_humour, axis=0).ravel()
            predicted_inter_humour =  np.stack(predicted_inter_humour, axis=0).ravel()
            
            target_inter_sarcasm = [t.cpu().numpy() for t in target_total_sarcasm]
            predicted_inter_sarcasm = [t.cpu().numpy() for t in predicted_total_sarcasm]
            target_inter_sarcasm =  np.stack(target_inter_sarcasm, axis=0).ravel()
            predicted_inter_sarcasm =  np.stack(predicted_inter_sarcasm, axis=0).ravel()
            
            target_inter_offensive = [t.cpu().numpy() for t in target_total_offensive]
            predicted_inter_offensive = [t.cpu().numpy() for t in predicted_total_offensive]
            target_inter_offensive =  np.stack(target_inter_offensive, axis=0).ravel()
            predicted_inter_offensive =  np.stack(predicted_inter_offensive, axis=0).ravel()
            
            target_inter_motivation = [t.cpu().numpy() for t in target_total_motivation]
            predicted_inter_motivation = [t.cpu().numpy() for t in predicted_total_motivation]
            target_inter_motivation =  np.stack(target_inter_motivation, axis=0).ravel()
            predicted_inter_motivation =  np.stack(predicted_inter_motivation, axis=0).ravel()
            
            
            
            batch_wise_loss.append(loss_average.item())
            
            batch_wise_micro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="micro"))
            batch_wise_macro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="macro"))
            
            batch_wise_micro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro"))
            batch_wise_macro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro"))
            
            batch_wise_micro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="micro"))
            batch_wise_macro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="macro"))
            
            batch_wise_micro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="micro"))
            batch_wise_macro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="macro"))
            
            
            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{CFG.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_average.item():.4f}')
                print(f' Humour Micro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_humour, predicted_inter_humour, average="micro")}')
                print(f' Humour Macro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_humour, predicted_inter_humour, average="macro")}')
                
                print(f' Sarcasm Micro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro")}')
                print(f' Sarcasm Macro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro")}')
                
                print(f' Offensive Micro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_offensive, predicted_inter_offensive, average="micro")}')
                print(f' Offensive Macro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_offensive, predicted_inter_offensive, average="macro")}')
            
                print(f' Motivation Micro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_motivation, predicted_inter_motivation, average="micro")}')
                print(f' Motivation Macro F1 on the training set after batch no {i+1}, \
                            Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter_motivation, predicted_inter_motivation, average="macro")}')
                
                
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
            epoch_wise_average_macro_f1.append(current_macro_average)
            
            epoch_wise_micro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="micro"))
            epoch_wise_macro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="macro"))
            
            epoch_wise_micro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro"))
            epoch_wise_macro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro"))
            
            epoch_wise_micro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="micro"))
            epoch_wise_macro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="macro"))
            
            epoch_wise_micro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="micro"))
            epoch_wise_macro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="macro"))

            
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
            
            print(f'Best Macro F1 on test set till this epoch: {max(epoch_wise_average_macro_f1)}, \
                         Found in Epoch No: {epoch_wise_average_macro_f1.index(max(epoch_wise_average_macro_f1))+1}')
            if best_wise_average_macro_f1 < current_macro_average:
                best_wise_average_macro_f1 = current_macro_average
                torch.save({'model': model.state_dict(), 
                            'classifier': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict()
                        },
                        f'onlyimage_fold{train_fold}_emotion_best.pth')
        
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()


###--- LOAD DATA-----------###
traindata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_train.csv', header=None)
testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 
#--train data---
train_images = traindata[:][0].to_numpy()[1:] # image paths
x = traindata[:][2].to_numpy()[1:] # text

humour_y = traindata[:][3].to_numpy()[1:]
sarcasm_y = traindata[:][4].to_numpy()[1:]
offensive_y = traindata[:][5].to_numpy()[1:]
motivational_y = traindata[:][6].to_numpy()[1:]
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

humour_y, sarcasm_y, offensive_y, motivational_y = textlabel_to_digitallabel(humour_y, sarcasm_y, offensive_y, motivational_y)
humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest = textlabel_to_digitallabel(humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest)

# train fold
y = [] # aggregated labels such as: [[0,0, 1, 1], [0,1, 0, 1], , [1,1, 1, 1], [1,0, 1,0], [1,0, 1, 1]]
for i in range(len(humour_y)):
    y.append([humour_y[i], sarcasm_y[i], offensive_y[i], motivational_y[i]])

for train_fold in CFG.trn_fold:
    folds = MultilabelStratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed).split(train_images, y)
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold == train_fold:  
            train_loop(trn_idx, val_idx)

