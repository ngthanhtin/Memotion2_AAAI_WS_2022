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
from utils.dataset_unimodal import MemoDataset_Sentiment
from utils.tokenizer import Tokenizer
# import models
from models.model_cnnbert import CNN_Roberta_Concat, CNN_Roberta_SAN, CNN_Roberta_Concat_HybridFusion, CNN_Roberta_Discrete
from transformers import BertTokenizer, RobertaTokenizer
###
from models.classifier import ClassifierLSTM_Sentiment
from utils.config import CFG
from utils.clean_text import *
from utils.radam.radam import RAdam
from utils.lookahead.optimizer import Lookahead
###
from utils.multiplicative_loss import Multiplicative_CrossEntropy, Multiplicative_CrossEntropy_V2
# manually fix batch size
CFG.batch_size = 10
CFG.model_name = 'cnnbert_discrete'
CFG.learning_rate = 2e-5
CFG.device = 'cuda:2'

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
    print('Training with fold {} started'.format(train_fold))

    train_data = MemoDataset_Sentiment(np.array(train_images), x, target_tensor, CFG.train_path, \
        roberta_tokenizer, CFG.max_len, transform=get_transforms(data = 'train')) 
    test_data = MemoDataset_Sentiment(np.array(test_images), xtest, target_tensor_test, CFG.test_path, \
        roberta_tokenizer, CFG.max_len, transform=None) 
    # sampler = sampler,
    trainloader = DataLoader(train_data, batch_size=CFG.batch_size,  shuffle=True, drop_last = True, num_workers=4) # if have sampler, dont use shuffle
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
    
    model = CNN_Roberta_Discrete(roberta_model_name = 'distilroberta-base', cnn_type = CFG.cnn_type, num_classes=CFG.n_sentiment_classes)


    # states = torch.load(f'{CFG.model_name}_fold0_sentiment_best.pth', map_location = torch.device('cpu'))
    # model.load_state_dict(states['model'])
    model.to(CFG.device)
    params = list(model.parameters())

    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient.to(device)) #  weight_class

    # criterion = nn.CrossEntropyLoss() #  weight_class 
    beta = 0.5
    criterion = Multiplicative_CrossEntropy(3, 3, beta, device=CFG.device)
    ce = nn.CrossEntropyLoss()
    # criterion = compute_cmpc_loss
    # dcca_criterion = model.dcca_criterion
    #
    base_optimizer = RAdam(params, lr = CFG.learning_rate, weight_decay=CFG.weight_decay)
    optimizer = Lookahead(optimizer = base_optimizer, k = 5, alpha=0.5)

    scheduler = get_scheduler(optimizer)

    batch_wise_loss = []
    batch_wise_micro_f1 = []
    batch_wise_macro_f1 = []
    epoch_wise_macro_f1 = []
    epoch_wise_micro_f1 = []

    best_acc = 0.
    min_loss = np.Inf

    best_f1 = 0.0
    best_f1_mavg = 0.0

    # Early Stopping
    n_epochs_stop = 6
    epochs_no_improve = 0
    early_stop = False
    # Train the model
    n_total_steps = len(trainloader)
   
    for epoch in range(CFG.epochs):
        
        target_total = []
        predicted_total = []
        
        for i, batch_dict in enumerate(trainloader):  
            model.train()

            indices = batch_dict['x_indices'].to(CFG.device)
            attn_mask = batch_dict['x_attn_mask'].to(CFG.device)
            images = batch_dict['x_images'].to(CFG.device)
            labels = batch_dict['y_target'].to(CFG.device)

            outputs = model(indices, attn_mask, images)
            loss, outputs_new = criterion(outputs, labels)
            outputs = outputs[-1] * outputs_new
            loss = ce(outputs, labels)
            # loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            
            target_total.append(labels)
            predicted_total.append(predicted)
            
            target_inter = [t.cpu().numpy() for t in target_total]
            predicted_inter = [t.cpu().numpy() for t in predicted_total]
            target_inter =  np.stack(target_inter, axis=0).ravel()
            predicted_inter =  np.stack(predicted_inter, axis=0).ravel()
            
            batch_wise_loss.append(loss.item())
            batch_wise_micro_f1.append(f1_score(target_inter, predicted_inter, average="micro"))
            batch_wise_macro_f1.append(f1_score(target_inter, predicted_inter, average="macro"))
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{CFG.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}') # DCCALoss: {dcca_loss.item():.4f} 
                print(f' Micro F1 on the training set after batch no {i+1}, \
                Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter, predicted_inter, average="micro")}')
                print(f' Macro F1 on the training set after batch no {i+1}, \
                Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter, predicted_inter, average="macro")}')
                print(confusion_matrix(target_inter, predicted_inter))
                # print(image_precision, text_precision)
            
            if min_loss > loss.item():
                min_loss = loss.item()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epoch > 5 and epochs_no_improve == n_epochs_stop:
                print('Early stopping!' )
                early_stop = True
                break
            else:
                continue
        # Check early stopping condition
        if early_stop:
            print("Stopped")
            break

        target_total_test = []
        predicted_total_test = [] 

        # Test the model
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for i_batch, batch_dict in enumerate(testloader):
                model.eval()

                indices = batch_dict['x_indices'].to(CFG.device)
                attn_mask = batch_dict['x_attn_mask'].to(CFG.device)
                images = batch_dict['x_images'].to(CFG.device)
                labels = batch_dict['y_target'].to(CFG.device)

                outputs = model(indices, attn_mask, images)
                _, outputs_new = criterion(outputs, labels)
                outputs = outputs[-1] * outputs_new
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
            epoch_wise_macro_f1.append(f1_score(target_inter, predicted_inter, average="macro"))
            epoch_wise_micro_f1.append(f1_score(target_inter, predicted_inter, average="micro"))

            acc = 100.0 * n_correct / n_samples

            f1_mavg = sum(epoch_wise_macro_f1) / len(epoch_wise_macro_f1)

            if current_macro > best_f1:
                best_f1 = current_macro
                print("Save model....")
                torch.save({'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict()
                        },
                        f'{CFG.model_name}_fold{train_fold}_sentiment_best.pth')
                
            if f1_mavg > best_f1_mavg:
                best_f1_mavg = f1_mavg
            if acc > best_acc:
                best_acc = acc
                
            print(f'Accuracy of the network on the test set after Epoch {epoch+1} is: {acc} %')        
            print(f' Micro F1 on the testing: {f1_score(target_inter, predicted_inter, average="micro")}')
            print(f' Macro F1 on the testing: {f1_score(target_inter, predicted_inter, average="macro")}')
            print(confusion_matrix(target_inter, predicted_inter))   
            print(f'Best Macro F1 on test set till this epoch: {max(epoch_wise_macro_f1)} \
            Found in Epoch No: {epoch_wise_macro_f1.index(max(epoch_wise_macro_f1))+1}')
            

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

### --- LOAD TOKENIZER----- ###
roberta_tokenizer = tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base', do_lower_case=True)

###----- LOAD DATA---------###
dataFrame = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_train.csv', header=None)
testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 

train_images = dataFrame[:][0].to_numpy()[1:] # image paths
x = dataFrame[:][2].to_numpy()[1:] # text
y = dataFrame[:][7].to_numpy()[1:] # label

#-----DELETE NON WORD INDEXES-----#
# 545, 3145 is no-word indexes
train_images = np.delete(train_images, 545)
train_images = np.delete(train_images, 3144)
x = np.delete(x, 545)
x = np.delete(x, 3144)
y = np.delete(y, 545)
y = np.delete(y, 3144)
######
test_images = testdata[:][0].to_numpy()[1:]
xtest = testdata[:][1].to_numpy()[1:]
ytest = testdata[:][7].to_numpy()[1:]

def calculateWeights(ylabel):
    negative_occurrences = ylabel.count(0)
    neutral_occurrences = ylabel.count(1)
    positive_occurrences = ylabel.count(2)
    weight_class = []
    weight_class.append(len(ylabel)/negative_occurrences)
    weight_class.append(len(ylabel)/neutral_occurrences)
    weight_class.append(len(ylabel)/positive_occurrences)
    
    weights = []
    for i in range(len(ylabel)):
        weights.append(weight_class[ylabel[i]])
    return weights, weight_class

#------ENCODER LABEL---------
"""
For train
negative: 973
neutral is largest: 4510
positive: 1517
For validation
0: negative, 1: neutral, 2: positive
negative: 200
neutral is largest: 975
positive: 325

"""
labelencode = LabelEncoder()
ylabel = labelencode.fit_transform(y.astype(str))
target_tensor = (ylabel.tolist())
# print(labelencode.inverse_transform([0, 0, 1, 2]))

ylabeltest = labelencode.fit_transform(ytest.astype(str))
target_tensor_test = (ylabeltest.tolist())

#-------SCRUB WORDS-----------------
x = [element.lower() if isinstance(element, str) else element for element in x]
xtest = [element.lower() if isinstance(element, str) else element for element in xtest]

x = [scrub_words(element) if isinstance(element, str) else element for element in x]
xtest = [scrub_words(element) if isinstance(element, str) else element for element in xtest]
print("Finish scrubing words...")

train_loop(0, 0)
# train fold
# for train_fold in CFG.trn_fold:
#     folds = StratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed).split(np.arange(len(train_images)), y)
#     for fold, (trn_idx, val_idx) in enumerate(folds):
#         if fold == train_fold:  
#             train_loop(trn_idx, val_idx)

