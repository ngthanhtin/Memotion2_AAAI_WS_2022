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
CFG.batch_size = 4

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

def train_loop(trn_idx, val_idx):
    train_fold = 0
    # print('Training with fold {} started'.format(train_fold))
    # #train
    # train_input_tensor_pad = [input_tensor_pad[index] for index in trn_idx]
    # train_humour = [humour_y[index] for index in trn_idx]
    # train_sarcasm = [sarcasm_y[index] for index in trn_idx]
    # train_offensive = [offensive_y[index] for index in trn_idx]
    # train_motivational = [motivational_y[index] for index in trn_idx]
    # #val
    # val_input_tensor_pad = [input_tensor_pad[index] for index in val_idx]
    # val_humour = [humour_y[index] for index in val_idx]
    # val_sarcasm = [sarcasm_y[index] for index in val_idx]
    # val_offensive = [offensive_y[index] for index in val_idx]
    # val_motivational = [motivational_y[index] for index in val_idx]

    # train_data = MemoDataset_Emotion(train_images[trn_idx], train_input_tensor_pad, train_humour, train_sarcasm, train_offensive, train_motivational, \
    #                                                 root_dir = CFG.train_path, transform=get_transforms(data = 'train'), task='emotion')
    # test_data = MemoDataset_Emotion(train_images[val_idx], val_input_tensor_pad, val_humour, val_sarcasm, val_offensive, val_motivational, \
    #                                                 root_dir = CFG.train_path, transform=None, task='emotion') 

    train_data = MemoDataset_Emotion(train_images, input_tensor_pad, humour_y, sarcasm_y, offensive_y, motivational_y, \
                                                    root_dir = CFG.train_path, transform=get_transforms(data = 'train'), task='emotion')
    test_data = MemoDataset_Emotion(test_images, input_tensor_test_pad, humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest, \
                                                    root_dir = CFG.test_path, transform=None, task='emotion') 
    
    trainloader = DataLoader(train_data, batch_size=CFG.batch_size, drop_last = True, shuffle=True, num_workers=4)    
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

    ### Model
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

    model.to(CFG.device)
    classifier.to(CFG.device)
    params = list(model.parameters()) + list(classifier.parameters())


    # Loss and optimizer
    # criterion_humour = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_humour)
    # criterion_sarcasm = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_sarcasm)
    # criterion_offensive = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_offensive)
    # criterion_motivation = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient_motivation)

    criterion_humour = nn.CrossEntropyLoss()
    criterion_sarcasm = nn.CrossEntropyLoss()
    criterion_offensive = nn.CrossEntropyLoss()
    criterion_motivation = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params, lr = CFG.learning_rate, weight_decay=CFG.weight_decay)
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
    for epoch in range(CFG.epochs):
        
        target_total_humour = []
        predicted_total_humour = []
        
        target_total_sarcasm = []
        predicted_total_sarcasm = []
        
        target_total_offensive = []
        predicted_total_offensive = []
        
        target_total_motivation = []
        predicted_total_motivation = []
        
        for i, (images, input_ids, labels_humour, labels_sarcasm, labels_offensive, labels_motivation) in enumerate(trainloader):  
            
            model.train()
            classifier.train()
            # Forward pass
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
                        f'{CFG.model_name}_fold{train_fold}_emotion_best.pth')
        
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

### --- LOAD TOKENIZER----- ###
tokenizer = torch.load(CFG.tokenizer_path)

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

#-------SCRUB WORDS-----------------
x = [element.lower() if isinstance(element, str) else element for element in x]
xtest = [element.lower() if isinstance(element, str) else element for element in xtest]

x = [scrub_words(element) if isinstance(element, str) else element for element in x]
xtest = [scrub_words(element) if isinstance(element, str) else element for element in xtest]
print("Finish scrubing words...")

#---------VECTORIZE TENSOR-----------
input_tensor = []
for idx in range(len(x)):
    input_tensor_sample = [tokenizer.stoi.get(s) for s in str(x[idx]).split(' ')]
    input_tensor.append(input_tensor_sample)

input_tensor_test = []
for idx in range(len(xtest)):
    input_tensor_testsample = [tokenizer.stoi.get(s) for s in str(xtest[idx]).split(' ')]
    input_tensor_test.append(input_tensor_testsample)
print("Finish vectorizing tensor...")

#### REPRESENT OUT-OF-VOCAB WORD BY A SPACE ####
    
for idx in range(len(input_tensor)):
    for v in range(len(input_tensor[idx])):
        if (input_tensor[idx][v] == None):
            input_tensor[idx][v] = 1

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
input_tensor_pad = []
input_tensor_test_pad = []

for idx in range(len(input_tensor)):
    xsample = [pad_sequences(input_tensor[idx], CFG.max_len)]
    input_tensor_pad.append(xsample)

for idx in range(len(input_tensor_test)):
    xsampletest = [pad_sequences(input_tensor_test[idx], CFG.max_len)]
    input_tensor_test_pad.append(xsampletest)
print("Finish padding...")

# train fold
# y = [] # aggregated labels such as: [[0,0, 1, 1], [0,1, 0, 1], , [1,1, 1, 1], [1,0, 1,0], [1,0, 1, 1]]
# for i in range(len(humour_y)):
#     y.append([humour_y[i], sarcasm_y[i], offensive_y[i], motivational_y[i]])

# for train_fold in CFG.trn_fold:
#     folds = MultilabelStratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed).split(train_images, y)
#     for fold, (trn_idx, val_idx) in enumerate(folds):
#         if fold == train_fold:  
#             print(train_loop(trn_idx, val_idx))


train_loop(0,0)
        
        


    





