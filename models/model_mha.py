import torch
import torch.nn as nn

from torch.nn import functional as F

import numpy as np

from models.cnn import CNN
from utils.config import CFG

class MemoLSTM_MHA(nn.Module):
    """
    Bimodal includes CNN and LSTM, and use Multi-hop Attention network as Fusion module
    """
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, hidden_d, dropout, n_layers, cnn_type, device):
        super(MemoLSTM_MHA, self).__init__()
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.hidden_d = hidden_d
        self.dropout = dropout
        self.device = device
        self.n_layers = n_layers

        atmf_dense_1   = 256
        atmf_dense_2   = 64
        atmf_dense_3   = 8
        atmf_dense_4   = 1

        #Word embedding
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(np.load(CFG.word_embedding)))

        #sentence embedding
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=dropout, num_layers = n_layers, bidirectional=True)
        self.textual_fc = nn.Linear(512, 512)
        
        # image embedding
        self.cnn = CNN(is_pretrained=True, type_=cnn_type)
        self.visual_fc = nn.Linear(1792, 512) #
        self.drop = nn.Dropout(p=0.2)
        
        self.l_1 = nn.Linear(2*hidden_size*30, hidden_d)

        
        self.W_b = nn.Parameter(torch.rand(self.batch_size, 512, 512))
        
        self.W_s1 = nn.Linear(512, 350) #### 2*units = 512
        self.W_s2 = nn.Linear(350, 30)
        
        self.atmf_dense_1 = nn.Linear(hidden_d, atmf_dense_1)
        self.atmf_dense_2 = nn.Linear(atmf_dense_1, atmf_dense_2)
        self.atmf_dense_3 = nn.Linear(atmf_dense_2, atmf_dense_3)
        self.atmf_dense_4 = nn.Linear(atmf_dense_3, atmf_dense_4)
        self.W_F = nn.Parameter(torch.rand(batch_size, 512, 512))
        self.W_f = nn.Parameter(torch.rand(batch_size, 512, 1))
        

    def atmf(self, mm_feature):

        mm_feature_tran = mm_feature.permute(0,2,1)
        s = self.atmf_dense_4(self.atmf_dense_3(self.atmf_dense_2(self.atmf_dense_1(mm_feature_tran))))
        s = s.permute(0,2,1)
        
        s = F.softmax(s, dim=2) + 1
    
        wei_fea  = mm_feature * s
        P_F = torch.tanh(torch.bmm(self.W_F, wei_fea))
        P_F = F.softmax(P_F, dim=2)
        
        
        gamma_f = torch.bmm(self.W_f.permute(0,2,1),P_F)
        gamma_f = gamma_f.permute(0,2,1)
        atmf_output = torch.bmm(wei_fea,gamma_f)
        

        return atmf_output
    
     
    
    def attention_net(self, lstm_output):

        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix
    
    
    
    def image_encoding_filter(self, text_features, img_features):
        
        #### text_features - (batch_size, num_seq_n, 512)
        #### img_features - (batch_size, num_seq_m, 512)
        img_features_tran = img_features.permute(0, 2, 1)
        
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
                
    
    
    def forward(self, img, input_sentences):

        input_split = self.word_embeddings(input_sentences)
        input_split = input_split.permute(1,0,2)

        output, (h_n, c_n) = self.bilstm(input_split)
        output = output.permute(1, 0, 2)
        # output = self.textual_fc(output)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)

        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        hidden_matrix = hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])


        # img_features = self.vgg19bottom(img)
        # img_features = img_features.reshape(self.batch_size,512,49)
        # img_features = img_features.permute(0, 2, 1)
        img_features = self.cnn(img)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)

        img_features = self.visual_fc(img_features)
        
        img_features = self.image_encoding_filter(output, img_features)
        
        attn_weight_matrix = self.attention_net(img_features)
        atten_img_features = torch.bmm(attn_weight_matrix, img_features)
        atten_img_features = atten_img_features.view(-1, atten_img_features.size()[1] * atten_img_features.size()[2])

        # Let's now concatenate the hidden_matrix, apply atmf and connect it to the fully connected layer.
        s_text = self.l_1(hidden_matrix)
        s_image = self.l_1(atten_img_features)

        mm_features = torch.stack((s_text,s_image)).permute(1,2,0)
        

        atmf_output = self.atmf(mm_features)
        atmf_output = torch.squeeze(atmf_output,2)
        
        return atmf_output

        
        
