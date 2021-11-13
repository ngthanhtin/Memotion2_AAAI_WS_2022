
import torch
from torch.utils.data import Dataset

import os, cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.transformation import get_transforms
#### Define the custom dataset
class MemoDataset_Sentiment(Dataset):
    """
    Dataset for classify sentiment: positive, negative, neutral
    """

    def __init__(self, imagelist, input_ids, ylist, root_dir, transform=None):
        
        self.imagelist = imagelist
        self.input_ids = input_ids
        self.targetlist = ylist
        self.root_dir = root_dir
        self.transform = transform
        self.val_transform = get_transforms(data="valid")

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.imagelist[idx])
        try:
            image = Image.open(img_name + ".jpg").convert('RGB')
        except:
            image = Image.open(img_name + ".png").convert('RGB')
        
        target = self.targetlist[idx]
        input_id = self.input_ids[idx]
        
        if self.transform:
            if (target == 0 or target == 2):
                image = self.transform(image)
            else:
                image = self.val_transform(image)
        else:
            preproc_img = self.val_transform(image)

        return (image, input_id, target)

#### Define the custom dataset
class MemoDataset_Emotion(Dataset):
    """
    Dataset for classify emotion: humour, sarcasm, offensive, motivation
    """
    def __init__(self, imagelist, input_ids, y_humourlist, y_sarcasmlist, y_offensivelist, y_motivationlist, root_dir, transform=None, task='emotion'):
        self.task = task
        self.imagelist = imagelist
        self.input_ids = input_ids
        self.humourlist = y_humourlist
        self.sarcasmlist = y_sarcasmlist
        self.offensivelist = y_offensivelist
        self.motivationlist = y_motivationlist
        self.root_dir = root_dir
        self.transform = transform
        self.val_transform = get_transforms(data='valid')

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.imagelist[idx])
        try:
            image = Image.open(img_name + ".jpg").convert('RGB')
        except:
            image = Image.open(img_name + ".png").convert('RGB')
            
        input_id = self.input_ids[idx]
        
        target_humour = self.humourlist[idx]
        target_sarcasm = self.sarcasmlist[idx]
        target_offensive = self.offensivelist[idx]
        target_motivation = self.motivationlist[idx]
        

        if self.transform:
            if self.task == 'intensity':
                if target_humour == 0 or target_humour == 2 or target_humour == 3:
                    preproc_img = self.transform(image)
                if target_sarcasm == 1 or target_sarcasm == 2 or target_sarcasm == 3:
                    preproc_img = self.transform(image)
                if target_offensive == 1 or target_offensive == 2 or target_offensive == 3:
                    preproc_img = self.transform(image)
                if target_motivation == 1:
                    preproc_img = self.transform(image)
                else:
                    preproc_img = self.val_transform(image)
            if self.task == 'emotion':
                if target_humour == 0:
                    preproc_img = self.transform(image)
                if target_sarcasm == 0 or target_sarcasm == 1:
                    preproc_img = self.transform(image)
                if target_offensive == 1:
                    preproc_img = self.transform(image)
                if target_motivation == 1:
                    preproc_img = self.transform(image)
                else:
                    preproc_img = self.val_transform(image)
        else:
            preproc_img = self.val_transform(image)
          
        return (preproc_img, input_id, target_humour, target_sarcasm, target_offensive, target_motivation)