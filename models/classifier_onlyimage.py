"""
Classifiers for Task B and Task C for only Image
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


"""
Task B: Emotion
"""
class Classifier_Emotion(nn.Module):
    def __init__(self, hidden_d, hidden_d2, n_classes, dropout, use_cbam = False):
        super(Classifier_Emotion, self).__init__()
        self.use_cbam = use_cbam
        if use_cbam:
            self.ca = ChannelAttention(hidden_d)
            self.sa = SpatialAttention()
        else:
            self.ca = None
            self.sa = None

        self.hidden_d = hidden_d
        self.hidden_d2 = hidden_d2
        self.n_classes = n_classes
        self.dropout = dropout

        self.prev_out_humour = nn.Linear(hidden_d, hidden_d2)
        self.out_humour = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_sarcasm = nn.Linear(hidden_d, hidden_d2)
        self.out_sarcasm = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_offensive = nn.Linear(hidden_d, hidden_d2)
        self.out_offensive = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_motivation = nn.Linear(hidden_d, hidden_d2)
        self.out_motivation = nn.Linear(hidden_d2, n_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2)
        
    
    def forward(self, fc_input):
        """
        fc_input: B, C, W, H
        """
        fc_input = self.avgpool(fc_input)
        
        if self.use_cbam:
            fc_input = self.ca(fc_input) * fc_input
            fc_input = self.sa(fc_input) * fc_input
            
        fc_input = fc_input.view(fc_input.size(0), fc_input.size(1), fc_input.size(2) * fc_input.size(3))

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
class Classifier_Intensity(nn.Module):
    def __init__(self, hidden_d, hidden_d2, n_classes_humour, n_classes_sarcasm, n_classes_offensive, n_classes_motivation, dropout, use_cbam = False):
        super(Classifier_Intensity, self).__init__()
        self.use_cbam = use_cbam
        if use_cbam:
            self.ca = ChannelAttention(hidden_d)
            self.sa = SpatialAttention()
        else:
            self.ca = None
            self.sa = None

        self.hidden_d = hidden_d
        self.hidden_d2 = hidden_d2
        self.n_classes_humour = n_classes_humour
        self.n_classes_sarcasm = n_classes_sarcasm
        self.n_classes_offensive = n_classes_offensive
        self.n_classes_motivation = n_classes_motivation
        self.dropout = dropout

        self.prev_out_humour = nn.Linear(hidden_d, hidden_d2)
        self.out_humour = nn.Linear(hidden_d2, n_classes_humour)
        
        self.prev_out_sarcasm = nn.Linear(hidden_d, hidden_d2)
        self.out_sarcasm = nn.Linear(hidden_d2, n_classes_sarcasm)
        
        self.prev_out_offensive = nn.Linear(hidden_d, hidden_d2)
        self.out_offensive = nn.Linear(hidden_d2, n_classes_offensive)
        
        self.prev_out_motivation = nn.Linear(hidden_d, hidden_d2)
        self.out_motivation = nn.Linear(hidden_d2, n_classes_motivation)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2)
        
    
    def forward(self, fc_input):
        """
        fc_input: B, C, W, H
        """
        fc_input = self.avgpool(fc_input)
        
        if self.use_cbam:
            fc_input = self.ca(fc_input) * fc_input
            fc_input = self.sa(fc_input) * fc_input

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