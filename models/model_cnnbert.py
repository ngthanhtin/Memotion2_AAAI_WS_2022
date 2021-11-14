from models.model_san import StackedAttention
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaModel
from models.cnn import CNN

from utils.config import CFG

from torch.nn import functional as F

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    

class RobertaClasasificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.swish = MemoryEfficientSwish()
        
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
           
class CNN_Roberta_Concat(nn.Module):
    """
    Use CNN and BERT, and use concatenation as fusion module
    """
    def __init__(self,
        roberta_model_name = 'distilroberta-base',
        cnn_type = 'efficientnetv2-s',
        num_classes =  3
        ):
        super().__init__()
        # text
        self.roberta = RobertaModel.from_pretrained(
            roberta_model_name,
            num_labels = num_classes,
        )
        self.roberta_clf = RobertaClasasificationHead(self.roberta.config)
        
        # vision
        self.cnn = CNN(is_pretrained=False, type_=cnn_type, num_classes=num_classes)
        self.dropout = nn.Dropout(p=0.2)

        # self.swish = MemoryEfficientSwish()
        self.output_fc = nn.Linear(2*num_classes,num_classes)

    def forward(self, indices, attn_mask, images):
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]
        y_pred_roberta = self.roberta_clf(roberta_out)

        y_pred_cnn = self.cnn(images)
        y_pred_cnn = self.dropout(y_pred_cnn)
        
        combined_y_pred = torch.cat([y_pred_roberta, y_pred_cnn],dim=1)
        combined_y_pred = self.dropout(self.output_fc(combined_y_pred))
        
        return combined_y_pred

class CNN_Roberta_SAN(nn.Module):
    """
    Use CNN and BERT, and use attention as fusion module
    """
    def __init__(self,
        roberta_model_name = 'distilroberta-base',
        cnn_type = 'efficientnetv2-s',
        num_classes = 4, # is the no classes of Emotion
        device = 'cpu'
        ):
        super().__init__()
        # text
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_fc = nn.Linear(768, 256)
        # vision
        self.cnn = CNN(is_pretrained=False, type_=cnn_type)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.v_att = StackedAttention(num_stacks=2, img_feat_size=1792, ques_feat_size=CFG.max_len*256, att_size=2048, drop_ratio=0.2, device=device)

        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(2048, num_classes)

    def forward(self, indices, attn_mask, images):
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]

        roberta_out = self.roberta_fc(roberta_out)
        textual_features = roberta_out.view(roberta_out.size(0), -1)

        img_features = self.cnn(images)
        # img_features = self.avgpool(img_features)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        
        # Attention
        att = self.v_att(img_features, textual_features)

        out = self.drop(self.out(att))
        return out

class CNN_Roberta_Concat_Intensity(nn.Module):
    """
    Use CNN and BERT, and use concatenation as fusion module
    """
    def __init__(self,
        roberta_model_name = 'distilroberta-base',
        cnn_type = 'efficientnetv2-s',
        n_humour_classes=4, n_sarcasm_classes=4, n_offensive_classes=4, n_motivation_classes=2, device='cpu'
        ):
        super().__init__()
        # text
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_fc = nn.Linear(768, 256)
        # vision
        self.cnn = CNN(is_pretrained=False, type_=cnn_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.drop = nn.Dropout(p=0.2)

        self.max_len = CFG.max_len
        self.out_humour = nn.Linear(self.max_len*256 + 1792, n_humour_classes)
        self.out_sarcasm = nn.Linear(self.max_len*256 + 1792, n_sarcasm_classes)
        self.out_offensive = nn.Linear(self.max_len*256 + 1792, n_offensive_classes)
        self.out_motivation = nn.Linear(self.max_len*256 + 1792, n_motivation_classes)

    def forward(self, indices, attn_mask, images):
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]
        roberta_out = self.roberta_fc(roberta_out)
        roberta_out = roberta_out.view(roberta_out.size(0), -1)

        image_features = self.cnn(images)
        fc_input = self.avgpool(image_features)
        fc_input = fc_input.view(fc_input.size(0), -1)

        concat_features = torch.cat([roberta_out, fc_input], dim=1)
        concat_features = self.drop(concat_features)
        
        logits_humour = self.out_humour(concat_features)
        logits_sarcasm = self.out_sarcasm(concat_features)
        logits_offensive = self.out_offensive(concat_features)
        logits_motivation = self.out_motivation(concat_features)

        return (logits_humour, logits_sarcasm, logits_offensive, logits_motivation)

class CNN_Roberta_SAN_Intensity(nn.Module):
    """
    Use CNN and BERT, and use attention as fusion module
    """
    def __init__(self,
        roberta_model_name = 'distilroberta-base',
        cnn_type = 'efficientnetv2-s',
        n_humour_classes=4, n_sarcasm_classes=4, n_offensive_classes=4, n_motivation_classes=2, device='cpu'
        ):
        super().__init__()
        # text
        self.roberta = RobertaModel.from_pretrained(roberta_model_name, output_hidden_states=True)
        self.roberta_fc = nn.Linear(768, 256)
        # vision
        self.cnn = CNN(is_pretrained=False, type_=cnn_type)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #attention
        self.att_size = 2048
        self.v_att = StackedAttention(num_stacks=2, img_feat_size=1792, ques_feat_size=CFG.max_len*256, att_size=self.att_size, drop_ratio=0.2, device=device)

        self.drop = nn.Dropout(p=0.2)

        self.out_humour = nn.Linear(self.att_size, n_humour_classes)
        self.out_sarcasm = nn.Linear(self.att_size, n_sarcasm_classes)
        self.out_offensive = nn.Linear(self.att_size, n_offensive_classes)
        self.out_motivation = nn.Linear(self.att_size, n_motivation_classes)

    def forward(self, indices, attn_mask, images):
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]

        roberta_out = self.roberta_fc(roberta_out)
        textual_features = roberta_out.view(roberta_out.size(0), -1)

        img_features = self.cnn(images)
        # img_features = self.avgpool(img_features)
        img_features = img_features.view(img_features.size(0), img_features.size(1), -1)
        img_features = img_features.permute(0, 2, 1)
        
        # Attention
        att = self.v_att(img_features, textual_features)
        att = self.drop(att)

        logits_humour = self.out_humour(att)
        logits_sarcasm = self.out_sarcasm(att)
        logits_offensive = self.out_offensive(att)
        logits_motivation = self.out_motivation(att)

        return (logits_humour, logits_sarcasm, logits_offensive, logits_motivation)

