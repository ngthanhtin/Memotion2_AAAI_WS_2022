from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur, CenterCrop
    )
from albumentations.pytorch import ToTensorV2
from torchvision import transforms,models

# transformations
def get_transforms(*, data):
    if data == 'train':
        train_transform = transforms.Compose([
                            transforms.Resize(256),                    
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomRotation(0.2),
                            transforms.ToTensor(),                
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                            )])
        return train_transform

    elif data == 'valid':
        valid_transform = transforms.Compose([
                            transforms.Resize(256), 
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),                     
                            transforms.Normalize(                      
                            mean=[0.485, 0.456, 0.406],                
                            std=[0.229, 0.224, 0.225]                  
                            )])
        return valid_transform