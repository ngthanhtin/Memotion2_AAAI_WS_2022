import timm
import torch.nn as nn

#----------MODEL-----------------------------------#
class VGG19Bottom(nn.Module):
    def __init__(self, original_model):
        super(VGG19Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x

class CNN(nn.Module):
    def __init__(self, is_pretrained=False, type_='efficientnetv2-m', num_classes = None):
        super(CNN, self).__init__()
        # efficientnetv2_rw_s: (32, 1792, 7, 7) , efficientnetv2_rw_m: (32, 2152, 7, 7), swin: (32, 49, 1024)
        # efficientnetb2: (32, 1408, 7,7)

        self.type_ = type_
        self.num_classes = num_classes
        if type_ == 'efficientnetv2-s':
            if num_classes:
                self.e = timm.create_model('efficientnetv2_rw_s', num_classes = num_classes, pretrained = is_pretrained)
            else:
                self.e = timm.create_model('efficientnetv2_rw_s', pretrained = is_pretrained)
        if type_ == 'efficientnetv2-m':
            if num_classes:
                self.e = timm.create_model('efficientnetv2_rw_m', num_classes = num_classes, pretrained = is_pretrained)
            else:
                self.e = timm.create_model('efficientnetv2_rw_m', pretrained = is_pretrained)
        if type_ == 'efficientnetv1':
            if num_classes:
                self.e = timm.create_model('efficientnet_b4', num_classes = num_classes, pretrained = is_pretrained)
            else:
                self.e = timm.create_model('efficientnet_b4', pretrained = is_pretrained)
        
        for p in self.e.parameters():
            p.requires_grad = True#False

    def forward(self, image):
        #batch_size, C, H, W = image.shape
        if self.num_classes:
            x = self.e(image)
        else:
            x = self.e.forward_features(image)
        
        return x