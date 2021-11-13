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
CFG.learning_rate = 2e-5
CFG.device = 'cuda:1'
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
    # train_input_tensor_pad = [input_tensor_pad[index] for index in trn_idx]
    # train_target_tensor = [target_tensor[index] for index in trn_idx]
    # val_input_tensor_pad = [input_tensor_pad[index] for index in val_idx]
    # val_target_tensor = [target_tensor[index] for index in val_idx]
    # #create sampler
    # weights, weight_class = calculateWeights(target_tensor)
    # weights = torch.FloatTensor(weights)
    # weight_class = torch.FloatTensor(weight_class).to(CFG.device)
    # print('weights: ', weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=True)

    # train_data = MemoDataset_Sentiment(train_images[trn_idx], train_input_tensor_pad, train_target_tensor, root_dir = CFG.train_path, transform=get_transforms(data = 'train'))
    # test_data = MemoDataset_Sentiment(train_images[val_idx], val_input_tensor_pad, val_target_tensor, root_dir = CFG.train_path, transform=get_transforms(data = 'train')) 
    
    train_data = MemoDataset_Sentiment(train_images, input_tensor_pad, target_tensor, root_dir = CFG.train_path, transform=get_transforms(data = 'train'))
    test_data = MemoDataset_Sentiment(test_images, input_tensor_test_pad, target_tensor_test, root_dir = CFG.test_path, transform=None) 
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

    model.to(CFG.device)
    classifier.to(CFG.device)
    params = list(model.parameters()) + list(classifier.parameters())

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight = CFG.class_weight_gradient.to(CFG.device)) #  weight_class
    # criterion = CB_loss
    optimizer = torch.optim.Adam(params, lr = CFG.learning_rate, weight_decay=CFG.weight_decay)
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
        
        for i, (images, input_ids, labels) in enumerate(trainloader):  
            
            model.train()
            classifier.train()

            images = images.to(CFG.device)
            labels = labels.to(CFG.device)

            input_utt = torch.tensor(input_ids[0]).to(CFG.device)
            att_output = model(images, input_utt)

            lstm_input = torch.unsqueeze(att_output, 0)
            outputs = classifier(lstm_input)
            loss = criterion(outputs, labels)
            
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
                print (f'Epoch [{epoch+1}/{CFG.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                print(f' Micro F1 on the training set after batch no {i+1}, \
                Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter, predicted_inter, average="micro")}')
                print(f' Macro F1 on the training set after batch no {i+1}, \
                Epoch [{epoch+1}/{CFG.epochs}]: {f1_score(target_inter, predicted_inter, average="macro")}')
                print(confusion_matrix(target_inter, predicted_inter))
            
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
            for i_batch, (images, input_ids, labels) in enumerate(testloader):
                model.eval()
                classifier.eval()
                images = images.to(CFG.device)
                labels = labels.to(CFG.device)

                input_utt = torch.tensor(input_ids[0]).to(CFG.device)
                att_output = model(images, input_utt)

                lstm_input = torch.unsqueeze(att_output, 0)
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
            epoch_wise_macro_f1.append(f1_score(target_inter, predicted_inter, average="macro"))
            epoch_wise_micro_f1.append(f1_score(target_inter, predicted_inter, average="micro"))

            acc = 100.0 * n_correct / n_samples

            f1_mavg = sum(epoch_wise_macro_f1) / len(epoch_wise_macro_f1)

            if current_macro > best_f1:
                best_f1 = current_macro
            if f1_mavg > best_f1_mavg:
                best_f1_mavg = f1_mavg
            if acc > best_acc:
                best_acc = acc
                print("Save model....")
                torch.save({'model': model.state_dict(), 
                            'classifier': classifier.state_dict(),
                        'optimizer': optimizer.state_dict(), 
                        'scheduler': scheduler.state_dict()
                        },
                        f'{CFG.model_name}_fold{train_fold}_sentiment_best.pth')
                

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
tokenizer = torch.load(CFG.tokenizer_path)

###----- LOAD DATA---------###
dataFrame = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_train.csv', header=None)
testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 

train_images = dataFrame[:][0].to_numpy()[1:] # image paths
x = dataFrame[:][2].to_numpy()[1:] # text
y = dataFrame[:][7].to_numpy()[1:] # label

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

# max_len = 0
# for i in x:
#     if len(i) > max_len:
#         max_len = len(i)
# print(max_len)
# exit()
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
    if len(x) > max_len: 
        padded[:] = x[:max_len]
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

train_loop(0, 0)
# train fold
# for train_fold in CFG.trn_fold:
#     folds = StratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed).split(np.arange(len(train_images)), y)
#     for fold, (trn_idx, val_idx) in enumerate(folds):
#         if fold == train_fold:  
#             train_loop(trn_idx, val_idx)

