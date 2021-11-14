"""
Using Stacked Attention
"""

import torch
import torch.nn as nn

from torchvision import models
from torch.nn import functional as F

import timm

import numpy as np
from models.cnn import CNN
from utils.config import CFG

"""
This code is extended from Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang's repository.
https://github.com/jnhwkim/ban-vqa
This code is modified from ZCYang's repository.
https://github.com/zcyang/imageqa-san
"""

# Stacked Attention
class StackedAttention(nn.Module):
    def __init__(self, num_stacks, img_feat_size, ques_feat_size, att_size, drop_ratio, device):
        super(StackedAttention, self).__init__()

        self.device = device

        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.att_size = att_size
        self.drop_ratio = drop_ratio
        self.num_stacks = num_stacks
        self.layers = nn.ModuleList()

        self.dropout = nn.Dropout(drop_ratio)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.fc11 = nn.Linear(ques_feat_size, att_size, bias=True)
        self.fc12 = nn.Linear(img_feat_size, att_size, bias=True)
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        # self.W_b = nn.Parameter(torch.rand(10, att_size, att_size))

        for stack in range(num_stacks - 1):
            self.layers.append(nn.Linear(att_size, att_size, bias=True))
            self.layers.append(nn.Linear(img_feat_size, att_size, bias=True))
            self.layers.append(nn.Linear(att_size, 1, bias=True))

    def image_encoding_filter(self, text_features, img_features):
        
        #### text_features - (batch_size, num_seq_n, 512)
        #### img_features - (batch_size, num_seq_m, 512)
        img_features_tran = img_features.permute(0, 2, 1)
        text_features = text_features.unsqueeze(1)
    
        affinity_matrix_int = torch.bmm(text_features, self.W_b)
        
        affinity_matrix = torch.bmm(affinity_matrix_int, img_features_tran)
        
        affinity_matrix_sum = torch.sum(affinity_matrix, dim=1)
        affinity_matrix_sum = torch.unsqueeze(affinity_matrix_sum, dim=1)
        alpha_h = affinity_matrix/affinity_matrix_sum

        alpha_h_tran = alpha_h.permute(0,2,1)
        a_h = torch.bmm(alpha_h_tran, text_features)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        gates = (1 - cos(img_features.cpu(), a_h.cpu())).to(self.device)

        gated_image_features = a_h * gates[:, :, None]     
        
        return gated_image_features

    def forward(self, img_feat, ques_feat, v_mask=False):

        # Batch size
        B = ques_feat.size(0)

        # Stack 1
        # img_feat = self.image_encoding_filter(ques_feat, img_feat)
        ques_emb_1 = self.fc11(ques_feat)
        img_emb_1 = self.fc12(img_feat)
        
        # Compute attention distribution
        h1 = self.tanh(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)
        h1_emb = self.fc13(self.dropout(h1))
        # Mask actual bounding box sizes before calculating softmax
        if v_mask:
            mask = (0 == img_emb_1.abs().sum(2)).unsqueeze(2).expand(h1_emb.size())
            h1_emb.data.masked_fill_(mask.data, -float('inf'))

        p1 = self.softmax(h1_emb)

        #  Compute weighted sum
        img_att_1 = img_emb_1*p1
        weight_sum_1 = torch.sum(img_att_1, dim=1)

        # Combine with question vector
        u1 = ques_emb_1 + weight_sum_1

        # Other stacks
        us = []
        ques_embs = []
        img_embs  = []
        hs = []
        h_embs =[]
        ps  = []
        img_atts = []
        weight_sums = []

        us.append(u1)
        for stack in range(self.num_stacks - 1):
            # img_embs[-1] = self.image_encoding_filter(us[-1], img_feat)
            ques_embs.append(self.layers[3 * stack + 0](us[-1]))
            img_embs.append(self.layers[3 * stack + 1](img_feat))
            
            # Compute attention distribution
            hs.append(self.tanh(ques_embs[-1].view(B, -1, self.att_size) + img_embs[-1]))
            h_embs.append(self.layers[3*stack + 2](self.dropout(hs[-1])))
            # Mask actual bounding box sizes before calculating softmax
            if v_mask:
                mask = (0 == img_embs[-1].abs().sum(2)).unsqueeze(2).expand(h_embs[-1].size())
                h_embs[-1].data.masked_fill_(mask.data, -float('inf'))
            ps.append(self.softmax(h_embs[-1]))

            #  Compute weighted sum
            img_atts.append(img_embs[-1] * ps[-1])
            weight_sums.append(torch.sum(img_atts[-1], dim=1))

            # Combine with previous stack
            ux = us[-1] + weight_sums[-1]

            # Combine with previous stack by multiple
            us.append(ux)

        return us[-1]

class MemoLSTM_SAN(nn.Module):
    """
    Bimodal includes CNN and LSTM, and use Stacked Attention network as Fusion module
    """
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, hidden_d, dropout, n_layers, cnn_type, device):
        super(MemoLSTM_SAN, self).__init__()
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.hidden_d = hidden_d
        self.dropout = dropout
        self.device = device
        self.n_layers = n_layers

        #Word embedding
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(np.load(CFG.word_embedding)))

        #sentence embedding
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=dropout, num_layers = n_layers, bidirectional=True)
        
        # image embedding
        self.cnn = CNN(is_pretrained=True, type_=cnn_type)
        self.drop = nn.Dropout(p=0.2)
        
        # attention
        self.v_att = StackedAttention(num_stacks=1, img_feat_size=1792, ques_feat_size=512, att_size=2048, drop_ratio=0.4, device=self.device)
        self.fc_out = nn.Linear(2048, 512)

    def forward(self, img, input_sentences):

        input_split = self.word_embeddings(input_sentences)
        input_split = input_split.permute(1,0,2)
        
        output, (h_n, c_n) = self.bilstm(input_split)
        output = h_n.permute(1, 0, 2)
        output = output.reshape(output.size(0), -1)

        img_features = self.cnn(img)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)

        # Attention
        att = self.v_att(img_features, output)
        fc_out = self.drop(self.fc_out(att))

        return fc_out


