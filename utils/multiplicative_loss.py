import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
import numpy as np

class Multiplicative_CrossEntropy(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax()
    softmax = nn.Softmax(dim=1)

    def __init__(self, num_multimodals, num_classes, beta, class_weights=None, device='cpu'):
        super().__init__()
        self.num_multimodals = num_multimodals
        self.num_classes = num_classes
        self.beta = beta
        if class_weights:
            self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).to(device))
        else:
            self.class_weights = None
        
        

    def forward(self, logits_m, target):
        """
        logits_m: m (multimodalities) logits
        """
        self.sum_log_probabilities = torch.zeros_like(logits_m[0])
        
        for i in range(len(logits_m)):
            logits = logits_m[i]
            probabilities = self.softmax(logits)
            weighting_factor = torch.pow(probabilities, self.beta/(self.num_multimodals-1))
            log_probabilities = self.log_softmax(logits)
            final_log_probabilities = weighting_factor * log_probabilities
            self.sum_log_probabilities += final_log_probabilities
        
        # # NLLLoss(x, class) = -weights[class] * x[class]
        if self.class_weights:
            return -self.class_weights.index_select(0, target) * self.sum_log_probabilities.index_select(-1, target).diag(), self.sum_log_probabilities
        return -self.sum_log_probabilities.index_select(-1, target).diag(), self.sum_log_probabilities


class Multiplicative_CrossEntropy_V2(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    log_softmax = nn.LogSoftmax()
    softmax = nn.Softmax()

    def __init__(self, num_multimodals, num_classes, beta, class_weights=None, device='cpu'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.num_multimodals = num_multimodals
        self.num_classes = num_classes
        self.beta = beta
        if class_weights:
            self.class_weights = autograd.Variable(torch.FloatTensor(class_weights).to(device))
        else:
            self.class_weights = None
        
        

    def forward(self, logits_m, target):
        """
        logits_m: m (multimodalities) logits
        """
        self.sum_log_probabilities = torch.zeros_like(logits_m[0])
        for i in range(len(logits_m)):
            logits = logits_m[i]
            probabilities = logits
            weighting_factor = torch.pow(probabilities, self.beta/(self.num_multimodals-1))
            log_probabilities = torch.log(logits)
            final_log_probabilities = weighting_factor * log_probabilities
            self.sum_log_probabilities += final_log_probabilities
        
        return self.criterion(self.sum_log_probabilities/3, target), self.sum_log_probabilities/3

"""

class Multiplicative_CrossEntropy(object):
    def __init__(self, num_multimodals, num_classes, beta, device='cpu'):
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.num_multimodals = num_multimodals
        self.num_classes = num_classes
        self.beta = beta
        self.device = device

    def forward(self, outputs_m, labels):
        batch_size = labels.shape[0]
        values_m = []
        outputs_new = torch.zeros_like(outputs_m[0])
        for i in range(self.num_multimodals):
            outputs = outputs_m[i]
            values, indices = torch.max(outputs, 1)
            values = torch.pow(values, self.beta/(self.num_multimodals - 1))*torch.log(values)
            values_m.append(values)

        final_outputs = 0
        for i in range(self.num_multimodals):
            final_outputs += values_m[i]
            for j in range(self.num_classes):
                outputs_new[:, j] -= values_m[i][:, j]

        loss = F.log_softmax(final_outputs, dim=1)
            
        return -loss/batch_size, outputs_new
"""