from models.model_san import StackedAttention
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaModel
from models.cnn import CNN

from utils.config import CFG

from torch.nn import functional as F

from utils.CCALoss import cca_loss, GCCA_loss

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
        self.cnn = CNN(is_pretrained=True, type_=cnn_type, num_classes=num_classes)
        self.dropout = nn.Dropout(p=0.2)

        # self.swish = MemoryEfficientSwish()
        self.output_fc = nn.Linear(2*num_classes,num_classes)

        # the size of the new space learned by the model (number of the new features)
        # outdim_size = 3
        # # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
        # # if one option does not work for a network or dataset, try the other one
        # use_all_singular_values = False
        # self.dcca_criterion = GCCA_loss

    def forward(self, indices, attn_mask, images):
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]
        y_pred_roberta = self.roberta_clf(roberta_out)

        #
        y_pred_cnn = self.cnn(images)
        y_pred_cnn = self.dropout(y_pred_cnn)
        
        #
        combined_y_pred = torch.cat([y_pred_roberta, y_pred_cnn],dim=1)
        combined_y_pred = self.dropout(self.output_fc(combined_y_pred))
        
        return combined_y_pred

class CNN_Roberta_Concat_HybridFusion(nn.Module):
    """
    Use CNN and BERT, and use concatenation as fusion module
    Hybrid Fusion known as Mid Fusion
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
        self.cnn = CNN(is_pretrained=True, type_=cnn_type, num_classes=num_classes)
        self.dropout = nn.Dropout(p=0.2)

        # self.swish = MemoryEfficientSwish()
        self.output_fc_1 = nn.Linear(2*num_classes,num_classes)
        self.output_fc_2 = nn.Linear(num_classes*3, num_classes)

    def forward(self, indices, attn_mask, images):
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]
        y_pred_roberta = self.roberta_clf(roberta_out)

        y_pred_cnn = self.cnn(images)
        y_pred_cnn = self.dropout(y_pred_cnn)
        
        concat_features_1 = torch.cat([y_pred_roberta, y_pred_cnn],dim=1)
        y_pred_1 = self.dropout(self.output_fc_1(concat_features_1))

        concat_features_2 = torch.cat([concat_features_1, y_pred_1], dim=1)
        y_pred_2 = self.dropout(self.output_fc_2(concat_features_2))

        return y_pred_2

class CNN_Roberta_Discrete(nn.Module):
    """
    Use CNN and BERT
    return 3 prediction, 2 of them are for single model prediction, the last one is for the fusion prediction
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
        self.cnn = CNN(is_pretrained=True, type_=cnn_type, num_classes=num_classes)
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
        
        concat_features = torch.cat([y_pred_roberta, y_pred_cnn], dim=1)
        y_pred_fusion = self.dropout(self.output_fc(concat_features))

        return y_pred_roberta, y_pred_cnn, y_pred_fusion


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
        self.cnn = CNN(is_pretrained=True, type_=cnn_type)
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
        self.cnn = CNN(is_pretrained=True, type_=cnn_type)
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

class CNN_Roberta_Concat_Intensity_HybridFusion(nn.Module):
    """
    Use CNN and BERT, and use concatenation as fusion module
    Hybrid Fusion
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
        self.max_len = CFG.max_len
        
        self.out_humour_text = nn.Linear(self.max_len*256, n_humour_classes)
        self.out_sarcasm_text = nn.Linear(self.max_len*256, n_sarcasm_classes)
        self.out_offensive_text = nn.Linear(self.max_len*256, n_offensive_classes)
        self.out_motivation_text = nn.Linear(self.max_len*256, n_motivation_classes)

        # vision
        self.cnn = CNN(is_pretrained=True, type_=cnn_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_humour_vision = nn.Linear(1792, n_humour_classes)
        self.out_sarcasm_vision = nn.Linear(1792, n_sarcasm_classes)
        self.out_offensive_vision = nn.Linear(1792, n_offensive_classes)
        self.out_motivation_vision = nn.Linear(1792, n_motivation_classes)
        #
        self.drop = nn.Dropout(p=0.2)

        
        # classifier 1
        self.out_humour = nn.Linear(self.max_len*256 + 1792, n_humour_classes)
        self.out_sarcasm = nn.Linear(self.max_len*256 + 1792, n_sarcasm_classes)
        self.out_offensive = nn.Linear(self.max_len*256 + 1792, n_offensive_classes)
        self.out_motivation = nn.Linear(self.max_len*256 + 1792, n_motivation_classes)

        #classifier 2
        self.out_humour_2 = nn.Linear(3*n_humour_classes, n_humour_classes)
        self.out_sarcasm_2 = nn.Linear(3*n_sarcasm_classes, n_sarcasm_classes)
        self.out_offensive_2 = nn.Linear(3*n_offensive_classes, n_offensive_classes)
        self.out_motivation_2 = nn.Linear(3*n_motivation_classes, n_motivation_classes)
        
    def forward(self, indices, attn_mask, images):
        #text
        roberta_out = self.roberta(
            input_ids = indices, 
            attention_mask =  attn_mask, 
        )[0]
        roberta_out = self.roberta_fc(roberta_out)
        roberta_out = roberta_out.view(roberta_out.size(0), -1)

        out_humour_text = self.out_humour_text(roberta_out)
        out_sarcasm_text = self.out_sarcasm_text(roberta_out)
        out_offensive_text = self.out_offensive_text(roberta_out)
        out_motivation_text = self.out_motivation_text(roberta_out)

        #vison
        image_features = self.cnn(images)
        image_features = self.avgpool(image_features)
        image_features = image_features.view(image_features.size(0), -1)

        out_humour_vision = self.out_humour_vision(image_features)
        out_sarcasm_vision = self.out_sarcasm_vision(image_features)
        out_offensive_vision = self.out_offensive_vision(image_features)
        out_motivation_vision = self.out_motivation_vision(image_features)

        #fusion 1
        concat_features = torch.cat([roberta_out, image_features], dim=1)
        concat_features = self.drop(concat_features)    
        #classifier 1
        logits_humour = self.out_humour(concat_features)
        logits_sarcasm = self.out_sarcasm(concat_features)
        logits_offensive = self.out_offensive(concat_features)
        logits_motivation = self.out_motivation(concat_features)


        # fusion 2
        # concat classifer 1 with different classifiers
        self.concat_humour = torch.cat([out_humour_text, out_humour_vision, logits_humour], dim=1)
        self.concat_sarcasm = torch.cat([out_sarcasm_text, out_sarcasm_vision, logits_sarcasm], dim=1)
        self.concat_offensive = torch.cat([out_offensive_text, out_offensive_vision, logits_offensive], dim=1)
        self.concat_motivation = torch.cat([out_motivation_text, out_motivation_vision, logits_motivation], dim=1)

        out_humour_2 = self.out_humour_2(self.concat_humour)
        out_sarcasm_2 = self.out_sarcasm_2(self.concat_sarcasm)
        out_offensive_2 = self.out_offensive_2(self.concat_offensive)
        out_motivation_2 = self.out_motivation_2(self.concat_motivation)

        return (out_humour_2, out_sarcasm_2, out_offensive_2, out_motivation_2)

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
        self.cnn = CNN(is_pretrained=True, type_=cnn_type)
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

