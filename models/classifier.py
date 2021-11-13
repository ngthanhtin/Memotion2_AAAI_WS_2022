"""
Classifiers for Task A, Task B, Task C
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Task A: Sentiment
"""
class ClassifierLSTM_Sentiment(nn.Module):
    def __init__(self, hidden_d, hidden_d2, dropout, n_layers, n_classes, device):
        super(ClassifierLSTM_Sentiment, self).__init__()
    
        self.hidden_d = hidden_d
        self.hidden_d2 = hidden_d2
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.device = device

        self.prev_out = nn.Linear(10*hidden_d, hidden_d2)
        self.out = nn.Linear(hidden_d2, n_classes)
        self.drop = nn.Dropout(p=0.2)
        
        self.W_s1_last = nn.Linear(hidden_d, 100) #### hidden_d = 300
        self.W_s2_last = nn.Linear(100, 10)
    
      
        
    def attention_net_last(self, lstm_output):

        attn_weight_matrix = self.W_s2_last(torch.tanh(self.W_s1_last(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix
    
    
    
    def forward(self, fc_input):

        fc_input = fc_input.permute(1,0,2)
        attn_weight_matrix = self.attention_net_last(fc_input)
        hidden_matrix = torch.bmm(attn_weight_matrix, fc_input)
        
        mixed_features = hidden_matrix.contiguous().view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])
        
        logits = self.prev_out(mixed_features)
        logits = self.drop(logits)
        logits = self.out(logits)
        logits = self.drop(logits)

        return logits

"""
Task B: Emotion
"""
class ClassifierLSTM_Emotion(nn.Module):
    def __init__(self, hidden_d, hidden_d2, n_classes, dropout, n_layers):
        super(ClassifierLSTM_Emotion, self).__init__()
        
        self.hidden_d = hidden_d
        self.hidden_d2 = hidden_d2
        self.n_classes = n_classes
        self.dropout = dropout
        self.n_layers = n_layers

        self.prev_out_humour = nn.Linear(hidden_d, hidden_d2)
        self.out_humour = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_sarcasm = nn.Linear(hidden_d, hidden_d2)
        self.out_sarcasm = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_offensive = nn.Linear(hidden_d, hidden_d2)
        self.out_offensive = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_motivation = nn.Linear(hidden_d, hidden_d2)
        self.out_motivation = nn.Linear(hidden_d2, n_classes)
        
        self.drop = nn.Dropout(p=0.2)
        
    
    def forward(self, fc_input):

        fc_input = fc_input.permute(1, 0, 2)
        mixed_features = fc_input.contiguous().view(-1, fc_input.size()[1] * fc_input.size()[2])
        
        logits_humour = self.prev_out_humour(mixed_features)
        logits_humour = self.drop(logits_humour)
        logits_humour = self.out_humour(logits_humour)
        logits_humour = self.drop(logits_humour)
        
        logits_sarcasm = self.prev_out_sarcasm(mixed_features)
        logits_sarcasm = self.drop(logits_sarcasm)
        logits_sarcasm = self.out_sarcasm(logits_sarcasm)
        logits_sarcasm = self.drop(logits_sarcasm)
        
        logits_offensive = self.prev_out_offensive(mixed_features)
        logits_offensive = self.drop(logits_offensive)
        logits_offensive = self.out_offensive(logits_offensive)
        logits_offensive = self.drop(logits_offensive)
        
        logits_motivation = self.prev_out_motivation(mixed_features)
        logits_motivation = self.drop(logits_motivation)
        logits_motivation = self.out_motivation(logits_motivation)
        logits_motivation = self.drop(logits_motivation)

        return (logits_humour, logits_sarcasm, logits_offensive, logits_motivation)

"""
Task C: Intensity of Emotions
"""
class ClassifierLSTM_Intensity(nn.Module):
    def __init__(self, hidden_d, hidden_d2, n_classes_humour, n_classes_sarcasm, n_classes_offensive, n_classes_motivation, dropout, n_layers):
        super(ClassifierLSTM_Intensity, self).__init__()
        
        self.hidden_d = hidden_d
        self.hidden_d2 = hidden_d2
        self.n_classes_humour = n_classes_humour
        self.n_classes_sarcasm = n_classes_sarcasm
        self.n_classes_offensive = n_classes_offensive
        self.n_classes_motivation = n_classes_motivation
        self.dropout = dropout
        self.n_layers = n_layers

        self.prev_out_humour = nn.Linear(hidden_d, hidden_d2)
        self.out_humour = nn.Linear(hidden_d2, n_classes_humour)
        
        self.prev_out_sarcasm = nn.Linear(hidden_d, hidden_d2)
        self.out_sarcasm = nn.Linear(hidden_d2, n_classes_sarcasm)
        
        self.prev_out_offensive = nn.Linear(hidden_d, hidden_d2)
        self.out_offensive = nn.Linear(hidden_d2, n_classes_offensive)
        
        self.prev_out_motivation = nn.Linear(hidden_d, hidden_d2)
        self.out_motivation = nn.Linear(hidden_d2, n_classes_motivation)
        
        self.drop = nn.Dropout(p=0.2)
        
    
    def forward(self, fc_input):

        fc_input = fc_input.permute(1, 0, 2)
        mixed_features = fc_input.contiguous().view(-1, fc_input.size()[1] * fc_input.size()[2])
        
        logits_humour = self.prev_out_humour(mixed_features)
        logits_humour = self.drop(logits_humour)
        logits_humour = self.out_humour(logits_humour)
        logits_humour = self.drop(logits_humour)
        
        logits_sarcasm = self.prev_out_sarcasm(mixed_features)
        logits_sarcasm = self.drop(logits_sarcasm)
        logits_sarcasm = self.out_sarcasm(logits_sarcasm)
        logits_sarcasm = self.drop(logits_sarcasm)
        
        logits_offensive = self.prev_out_offensive(mixed_features)
        logits_offensive = self.drop(logits_offensive)
        logits_offensive = self.out_offensive(logits_offensive)
        logits_offensive = self.drop(logits_offensive)
        
        logits_motivation = self.prev_out_motivation(mixed_features)
        logits_motivation = self.drop(logits_motivation)
        logits_motivation = self.out_motivation(logits_motivation)
        logits_motivation = self.drop(logits_motivation)

        return (logits_humour, logits_sarcasm, logits_offensive, logits_motivation)