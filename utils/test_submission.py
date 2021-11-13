import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from config import CFG

###--- LOAD DATA-----------###
testdata = pd.read_csv('/home/tinvn/TIN/MEME_Challenge/memotion2/memotion_val.csv', header=None) 
#--test data---
test_images = testdata[:][0].to_numpy()[1:] # image paths      
xtest = testdata[:][2].to_numpy()[1:] # text
humour_ytest = testdata[:][3].to_numpy()[1:]
sarcasm_ytest = testdata[:][4].to_numpy()[1:]
offensive_ytest = testdata[:][5].to_numpy()[1:]
motivational_ytest = testdata[:][6].to_numpy()[1:]
# sentiment
sentiment_ytest = testdata[:][7].to_numpy()[1:]

def textlabel_to_digitallabel(humour_arr, sarcasm_arr, offensive_arr, motivational_arr):
    for idx in range(len(humour_arr)):
        if (humour_arr[idx] == 'not_funny'):
            humour_arr[idx] = 0
        if (humour_arr[idx] == 'funny'):
            humour_arr[idx] = 1
        if (humour_arr[idx] == 'very_funny'):
            humour_arr[idx] = 2
        if (humour_arr[idx] =='hilarious'):
            humour_arr[idx] = 3

    for idx in range(len(sarcasm_arr)):
        if (sarcasm_arr[idx] == 'not_sarcastic'):
            sarcasm_arr[idx] = 0
        if (sarcasm_arr[idx] == 'little_sarcastic'):
            sarcasm_arr[idx] = 1
        if (sarcasm_arr[idx] == 'very_sarcastic'):
            sarcasm_arr[idx] = 2
        if (sarcasm_arr[idx] == 'extremely_sarcastic'):
            sarcasm_arr[idx] = 3

    for idx in range(len(offensive_arr)):
        if (offensive_arr[idx] == 'not_offensive'):
            offensive_arr[idx] = 0
        if (offensive_arr[idx] == 'slight'):
            offensive_arr[idx] = 1
        if (offensive_arr[idx] == 'very_offensive'):
            offensive_arr[idx] = 2
        if (offensive_arr[idx] =='hateful_offensive'):
            offensive_arr[idx] = 3    
            
    for idx in range(len(motivational_arr)):
        if (motivational_arr[idx] == 'not_motivational'):
            motivational_arr[idx] = 0
        if (motivational_arr[idx] == 'motivational'):
            motivational_arr[idx] = 1 

    return humour_arr, sarcasm_arr, offensive_arr, motivational_arr

humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest = textlabel_to_digitallabel(humour_ytest, sarcasm_ytest, offensive_ytest, motivational_ytest)


f = open('answer.txt', 'w')
for i, row in testdata.iterrows():
    if i == 0:
        continue
    sentiment = sentiment_ytest[i - 1] # because ytest start with 0, but the original csv start with 1
    if sentiment == 'neutral':
        sentiment = 1
    if sentiment == 'negative' or sentiment == 'very_negative':
        sentiment = 0
    if sentiment == 'positive' or sentiment == 'very_positive':
        sentiment = 2
    humour = 0 if humour_ytest[i-1] == 0 else 1
    sarcasm = 0 if sarcasm_ytest[i-1] == 0 else 1
    offensive = 0 if offensive_ytest[i-1] == 0 else 1
    motivational = 0 if motivational_ytest[i-1] == 0 else 1

    intensity_humour = humour_ytest[i-1]
    intensity_sarcasm = sarcasm_ytest[i-1]
    intensity_offensive = offensive_ytest[i-1]
    intensity_motivational = motivational_ytest[i-1]

    f.write("{}_{}{}{}{}_{}{}{}{}\n".format(sentiment, humour, sarcasm, offensive, motivational, \
                                            intensity_humour, intensity_sarcasm, intensity_offensive, intensity_motivational))
                        


    
    






        
        


    





