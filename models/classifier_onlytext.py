"""
Classifiers for Task C for only Text
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import RobertaForSequenceClassification
"""
Task C: Intensity of Emotions
"""
class Classifier_Intensity(nn.Module):
    def __init__(self, n_classes_humour, n_classes_sarcasm, n_classes_offensive, n_classes_motivation, dropout):
        super(Classifier_Intensity, self).__init__()

        self.model_humour = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=n_classes_humour)
        self.model_sarcasm = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=n_classes_sarcasm)
        self.model_offensive = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=n_classes_offensive)
        self.model_motivation = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=n_classes_motivation)

        self.n_classes_humour = n_classes_humour
        self.n_classes_sarcasm = n_classes_sarcasm
        self.n_classes_offensive = n_classes_offensive
        self.n_classes_motivation = n_classes_motivation
        self.dropout = dropout
    
    def forward(self, indices, attn_mask):
        ouput_humour = self.model_humour(indices, token_type_ids=None, attention_mask=attn_mask)
        output_sarcasm = self.model_sarcasm(indices, token_type_ids=None, attention_mask=attn_mask)
        output_offensive = self.model_offensive(indices, token_type_ids=None, attention_mask=attn_mask)
        output_motivation = self.model_motivation(indices, token_type_ids=None, attention_mask=attn_mask)

        return (ouput_humour, output_sarcasm, output_offensive, output_motivation)