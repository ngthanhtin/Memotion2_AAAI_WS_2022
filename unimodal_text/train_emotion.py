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
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils.transformation import get_transforms
from utils.dataset_unimodal import MemoDataset_Emotion
from utils.tokenizer import Tokenizer
from transformers import BertTokenizer, RobertaTokenizer,RobertaModel, XLNetTokenizer, RobertaTokenizer, \
    BertForSequenceClassification, XLNetForSequenceClassification, RobertaModel, AdamW, RobertaForSequenceClassification
from utils.config import CFG
from utils.clean_text import *

from utils.radam.radam import RAdam
from utils.lookahead.optimizer import Lookahead

import utils.memotion_utils.general as general_utils
import utils.memotion_utils.transformer.data as transformer_data_utils
import utils.memotion_utils.transformer.general as transformer_general_utils
#-----

# manually fix CFG
CFG.max_len = 35
CFG.batch_size = 15
CFG.epochs = 10
CFG.learning_rate = 2e-5

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
    # train_input_tensor_pad = [x[index] for index in trn_idx]
    # train_humour = [humour_y[index] for index in trn_idx]
    # train_sarcasm = [sarcasm_y[index] for index in trn_idx]
    # train_offensive = [offensive_y[index] for index in trn_idx]
    # train_motivational = [motivational_y[index] for index in trn_idx]
    # #val
    # val_input_tensor_pad = [x[index] for index in val_idx]
    # val_humour = [humour_y[index] for index in val_idx]
    # val_sarcasm = [sarcasm_y[index] for index in val_idx]
    # val_offensive = [offensive_y[index] for index in val_idx]
    # val_motivational = [motivational_y[index] for index in val_idx]

    # train_data = MemoDataset_Emotion(None,  train_input_tensor_pad, train_humour, train_sarcasm, train_offensive, train_motivational, \
    #     CFG.train_path, roberta_tokenizer, CFG.max_len, transform=get_transforms(data = 'train')) 
    # test_data = MemoDataset_Emotion(None,  val_input_tensor_pad, val_humour, val_sarcasm, val_offensive, val_motivational, \
    #     CFG.train_path, roberta_tokenizer, CFG.max_len, transform=get_transforms(data = 'train')) 
    
    train_data = MemoDataset_Emotion(None,  x, humour_y, sarcasm_y, offensive_y, motivational_y, \
        CFG.train_path, roberta_tokenizer, CFG.max_len, transform=get_transforms(data = 'train')) 
    test_data = MemoDataset_Emotion(None,  xtest, humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest, \
        CFG.test_path, roberta_tokenizer, CFG.max_len, transform=get_transforms(data = 'train')) 
    # 
    trainloader = DataLoader(train_data, batch_size=CFG.batch_size, shuffle=True, drop_last = True, num_workers=4) # if have sampler, dont use shuffle
    testloader = DataLoader(test_data, batch_size=5, drop_last=True, shuffle=False, num_workers=4)
    
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

    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=CFG.n_sentiment_classes)
    model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=4) # because it has 4 classes
    model.to(CFG.device)
    
    params = list(model.parameters())

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss() #weight = CFG.class_weight_gradient
    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(params, lr = CFG.learning_rate, weight_decay=CFG.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999))
    base_optimizer = RAdam(params, lr = CFG.learning_rate, weight_decay=CFG.weight_decay)
    optimizer = Lookahead(optimizer = base_optimizer, k = 5, alpha=0.5)
    early_stopping = transformer_general_utils.EarlyStopping(patience=4)
    train_state = general_utils.make_train_state()
    train_state.keys()
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
        
        for i, batch_dict in enumerate(trainloader):  
            model.train()

            indices = batch_dict['x_indices'].to(CFG.device)
            attn_mask = batch_dict['x_attn_mask'].to(CFG.device)
            labels_humour = batch_dict['y_humour'].to(CFG.device)
            labels_sarcasm = batch_dict['y_sarcasm'].to(CFG.device)
            labels_offensive = batch_dict['y_offensive'].to(CFG.device)
            labels_motivation = batch_dict['y_motivation'].to(CFG.device)
            
            labels = [] # aggregated labels such as: [[0,0, 1, 1], [0,1, 0, 1], , [1,1, 1, 1], [1,0, 1,0], [1,0, 1, 1]]
            for k in range(len(labels_humour)):
                labels.append([labels_humour[k], labels_sarcasm[k], labels_offensive[k], labels_motivation[k]])
            labels = torch.Tensor(labels).to(CFG.device)
            
            outputs = model(indices, token_type_ids=None, attention_mask=attn_mask)
            logits = outputs[0]
            
            # labels_one_hot = torch.nn.functional.one_hot(labels, 4).float().to(CFG.device)
            loss = criterion(logits, labels)
            pred_label = torch.sigmoid(logits).detach().cpu().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate Accuracy
            threshold = 0.50
            pred_bools = np.where(pred_label > threshold, 1., 0.)
            # max returns (value ,index)
            predicted_humour = pred_bools[:,0]
            predicted_sarcasm = pred_bools[:,1]
            predicted_offensive = pred_bools[:,2]
            predicted_motivation = pred_bools[:,3]
            
            target_total_humour.append(labels_humour)
            predicted_total_humour.append(predicted_humour)
            
            target_total_sarcasm.append(labels_sarcasm)
            predicted_total_sarcasm.append(predicted_sarcasm)
            
            target_total_offensive.append(labels_offensive)
            predicted_total_offensive.append(predicted_offensive)
            
            target_total_motivation.append(labels_motivation)
            predicted_total_motivation.append(predicted_motivation)
            
            target_inter_humour = [t.cpu().numpy() for t in target_total_humour]
            predicted_inter_humour = [t for t in predicted_total_humour]
            target_inter_humour =  np.stack(target_inter_humour, axis=0).ravel()
            predicted_inter_humour =  np.stack(predicted_inter_humour, axis=0).ravel()
            
            target_inter_sarcasm = [t.cpu().numpy() for t in target_total_sarcasm]
            predicted_inter_sarcasm = [t for t in predicted_total_sarcasm]
            target_inter_sarcasm =  np.stack(target_inter_sarcasm, axis=0).ravel()
            predicted_inter_sarcasm =  np.stack(predicted_inter_sarcasm, axis=0).ravel()
            
            target_inter_offensive = [t.cpu().numpy() for t in target_total_offensive]
            predicted_inter_offensive = [t for t in predicted_total_offensive]
            target_inter_offensive =  np.stack(target_inter_offensive, axis=0).ravel()
            predicted_inter_offensive =  np.stack(predicted_inter_offensive, axis=0).ravel()
            
            target_inter_motivation = [t.cpu().numpy() for t in target_total_motivation]
            predicted_inter_motivation = [t for t in predicted_total_motivation]
            target_inter_motivation =  np.stack(target_inter_motivation, axis=0).ravel()
            predicted_inter_motivation =  np.stack(predicted_inter_motivation, axis=0).ravel()
            
            
            batch_wise_loss.append(loss.item())
            batch_wise_micro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="micro"))
            batch_wise_macro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="macro"))
            
            batch_wise_micro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro"))
            batch_wise_macro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro"))
            
            batch_wise_micro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="micro"))
            batch_wise_macro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="macro"))
            
            batch_wise_micro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="micro"))
            batch_wise_macro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="macro"))
            
            if (i+1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{CFG.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
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
            for i_batch, batch_dict in enumerate(testloader):
                model.eval()

                indices = batch_dict['x_indices'].to(CFG.device)
                attn_mask = batch_dict['x_attn_mask'].to(CFG.device)
                labels_humour = batch_dict['y_humour'].to(CFG.device)
                labels_sarcasm = batch_dict['y_sarcasm'].to(CFG.device)
                labels_offensive = batch_dict['y_offensive'].to(CFG.device)
                labels_motivation = batch_dict['y_motivation'].to(CFG.device)

                labels = [] # aggregated labels such as: [[0,0, 1, 1], [0,1, 0, 1], , [1,1, 1, 1], [1,0, 1,0], [1,0, 1, 1]]
                for k in range(len(labels_humour)):
                    labels.append([labels_humour[k], labels_sarcasm[k], labels_offensive[k], labels_motivation[k]])
                labels = torch.Tensor(labels).to(CFG.device)

                outputs = model(input_ids=indices, token_type_ids=None, attention_mask=attn_mask)
                logits = outputs[0]
                
                # pred_label = torch.sigmoid(logits)
                pred_label = torch.sigmoid(logits).detach().cpu().numpy()

                # Calculate Accuracy
                threshold = 0.50
                pred_bools = np.where(pred_label > threshold, 1., 0.)
                # max returns (value ,index)
                predicted_humour = pred_bools[:,0]
                predicted_sarcasm = pred_bools[:,1]
                predicted_offensive = pred_bools[:,2]
                predicted_motivation = pred_bools[:,3]

                target_total_test_humour.append(labels_humour)
                predicted_total_test_humour.append(predicted_humour)
            
                target_total_test_sarcasm.append(labels_sarcasm)
                predicted_total_test_sarcasm.append(predicted_sarcasm)
            
                target_total_test_offensive.append(labels_offensive)
                predicted_total_test_offensive.append(predicted_offensive)
            
                target_total_test_motivation.append(labels_motivation)
                predicted_total_test_motivation.append(predicted_motivation)

                target_inter_humour = [t.cpu().numpy() for t in target_total_test_humour]
                predicted_inter_humour = [t for t in predicted_total_test_humour]
                target_inter_humour =  np.stack(target_inter_humour, axis=0).ravel()
                predicted_inter_humour =  np.stack(predicted_inter_humour, axis=0).ravel()
            
                target_inter_sarcasm = [t.cpu().numpy() for t in target_total_test_sarcasm]
                predicted_inter_sarcasm = [t for t in predicted_total_test_sarcasm]
                target_inter_sarcasm =  np.stack(target_inter_sarcasm, axis=0).ravel()
                predicted_inter_sarcasm =  np.stack(predicted_inter_sarcasm, axis=0).ravel()
            
                target_inter_offensive = [t.cpu().numpy() for t in target_total_test_offensive]
                predicted_inter_offensive = [t for t in predicted_total_test_offensive]
                target_inter_offensive =  np.stack(target_inter_offensive, axis=0).ravel()
                predicted_inter_offensive =  np.stack(predicted_inter_offensive, axis=0).ravel()
            
                target_inter_motivation = [t.cpu().numpy() for t in target_total_test_motivation]
                predicted_inter_motivation = [t for t in predicted_total_test_motivation]
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
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict()
                        },
                        f'onlytext_fold{train_fold}_emotion_best.pth')
            

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

### --- LOAD TOKENIZER----- ###
class RobertaPreprocessor():
    def __init__(self,transformer_tokenizer,sentence_detector):
        self.transformer_tokenizer = transformer_tokenizer
        self.sentence_detector = sentence_detector
        self.bos_token = transformer_tokenizer.bos_token
        self.sep_token = ' ' + transformer_tokenizer.sep_token + ' '
    def add_special_tokens(self, text):
        text = str(text)
        sentences = self.sentence_detector.tokenize(text)
        eos_added_text  = self.sep_token.join(sentences) 
        return self.bos_token +' '+ eos_added_text + ' ' + self.transformer_tokenizer.sep_token
        
roberta_tokenizer = tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=True)
punkt_sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
roberta_preproc = RobertaPreprocessor(roberta_tokenizer, punkt_sentence_detector)


# max_length = 35
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # tokenizer
# encodings = tokenizer.batch_encode_plus(ocr_texts,max_length=max_length,pad_to_max_length=True) # tokenizer's encoding method
# print('tokenizer outputs: ', encodings.keys())

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


#-----DELETE NON WORD INDEXES-----#
# 545, 3145 is no-word indexes
train_images = np.delete(train_images, 545)
train_images = np.delete(train_images, 3144)
x = np.delete(x, 545)
x = np.delete(x, 3144)
humour_y = np.delete(humour_y, 545)
humour_y = np.delete(humour_y, 3144)
sarcasm_y = np.delete(sarcasm_y, 545)
sarcasm_y = np.delete(sarcasm_y, 3144)
offensive_y = np.delete(offensive_y, 545)
offensive_y = np.delete(offensive_y, 3144)
motivational_y = np.delete(motivational_y, 545)
motivational_y = np.delete(motivational_y, 3144)

#-------SCRUB WORDS-----------------
x = [element.lower() if isinstance(element, str) else element for element in x]
xtest = [element.lower() if isinstance(element, str) else element for element in xtest]

x = [scrub_words(element) if isinstance(element, str) else element for element in x]
xtest = [scrub_words(element) if isinstance(element, str) else element for element in xtest]
print("Finish scrubing words...")


# # train fold
# y = [] # aggregated labels such as: [[0,0, 1, 1], [0,1, 0, 1], , [1,1, 1, 1], [1,0, 1,0], [1,0, 1, 1]]
# for i in range(len(humour_y)):
#     y.append([humour_y[i], sarcasm_y[i], offensive_y[i], motivational_y[i]])

# for train_fold in CFG.trn_fold:
#     folds = MultilabelStratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed).split(train_images, y)
#     for fold, (trn_idx, val_idx) in enumerate(folds):
#         if fold == train_fold:  
#             train_loop(trn_idx, val_idx)

train_loop(0, 0)
