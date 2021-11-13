
import torch
from torch.utils.data import Dataset

import os, cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.transformation import get_transforms
#### Define the custom dataset
from typing import Callable

class SimpleVectorizer():
    def __init__(self,tokenizer: Callable, max_seq_len: int):
        """
        Args:
            tokenizer (Callable): transformer tokenizer
            max_seq_len (int): Maximum sequence lenght 
        """
        self.tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def vectorize(self,text :str):
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False, #already added by preproc
            max_length = self._max_seq_len,
            pad_to_max_length = True,
            truncation=True
        )
        ids =  np.array(encoded['input_ids'], dtype=np.int64)
        attn = np.array(encoded['attention_mask'], dtype=np.int64)
        
        return ids, attn

class MemoDataset_Sentiment(Dataset):
    def __init__(self, imagelist, input_ids, ylist, root_dir, tokenizer, max_len, transform=None):

        self.imagelist = imagelist
        self.input_ids = input_ids
        self.targetlist = ylist
        self.root_dir = root_dir
        self.transform = transform
        self.val_transform = get_transforms(data="valid")
        self.tokenizer = tokenizer
        self._max_seq_length = max_len
            
        self.vectorizer = SimpleVectorizer(tokenizer, self._max_seq_length)

    def __len__(self):
        if self.input_ids == None:
            return len(self.imagelist)
        else:
            return len(self.input_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        target = self.targetlist[idx] # label

            
        if self.input_ids == None:
            # use only image

            img_name = os.path.join(self.root_dir,
                                    self.imagelist[idx])
            try:
                image = Image.open(img_name + ".jpg").convert('RGB')
            except:
                print('Except')
                image = Image.open(img_name + ".png").convert('RGB')

            if self.transform:
                if (target == 0 or target == 2):
                    preproc_img = self.transform(image)
                else:
                    preproc_img = self.val_transform(image)
            else:
                preproc_img = self.val_transform(image)

            return {
                'x_images': preproc_img,
                'x_index': idx,
                'y_target': target,
            }
        
        if self.imagelist is None:
            # use only text
            input_id = self.input_ids[idx] # text
            indices, attention_masks = self.vectorizer.vectorize(input_id)

            return {
                'x_indices': indices,
                'x_attn_mask': attention_masks,
                'x_index': idx,
                'y_target': target,
            }

        # use image and text
        input_id = self.input_ids[idx] # text
        indices, attention_masks = self.vectorizer.vectorize(input_id)

        img_name = os.path.join(self.root_dir,
                                self.imagelist[idx])
        try:
            image = Image.open(img_name + ".jpg").convert('RGB')
        except:
            print('Except')
            image = Image.open(img_name + ".png").convert('RGB')

        if self.transform:
            if (target == 0 or target == 2):
                preproc_img = self.transform(image)
            else:
                preproc_img = self.val_transform(image)
        else:
            preproc_img = self.val_transform(image)

        return {
            'x_images': preproc_img,
            'x_indices': indices,
            'x_attn_mask': attention_masks,
            'x_index': idx,
            'y_target': target,
        }
    
class MemoDataset_Emotion(Dataset):
    """
    Dataset for classify emotion: humour, sarcasm, offensive, motivation
    """
    def __init__(self, imagelist, input_ids, y_humourlist, y_sarcasmlist, y_offensivelist, y_motivationlist, root_dir, tokenizer, \
         max_len, transform=None, task="emotion"):
        
        self.task = task #identify which task to consider augmentation (emotion or itensity)
        self.imagelist = imagelist
        self.input_ids = input_ids
        self.humourlist = y_humourlist
        self.sarcasmlist = y_sarcasmlist
        self.offensivelist = y_offensivelist
        self.motivationlist = y_motivationlist
        self.root_dir = root_dir
        self.transform = transform
        self.val_transform = get_transforms(data='valid')
        self.tokenizer = tokenizer
        self._max_seq_length = max_len
            
        self.vectorizer = SimpleVectorizer(tokenizer, self._max_seq_length)

    def __len__(self):
        if self.input_ids == None:
            return len(self.imagelist)
        else:
            return len(self.input_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        target_humour = self.humourlist[idx]
        target_sarcasm = self.sarcasmlist[idx]
        target_offensive = self.offensivelist[idx]
        target_motivation = self.motivationlist[idx]
            
        if self.input_ids == None:
            # use only image

            img_name = os.path.join(self.root_dir,
                                    self.imagelist[idx])
            try:
                image = Image.open(img_name + ".jpg").convert('RGB')
            except:
                print('Except')
                image = Image.open(img_name + ".png").convert('RGB')

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

            return {
                'x_images': preproc_img,
                'x_index': idx,
                'y_humour': target_humour,
                'y_sarcasm': target_sarcasm,
                'y_offensive': target_offensive,
                'y_motivation': target_motivation
            }
        
        if self.imagelist is None:
            # use only text
            input_id = self.input_ids[idx] # text
            indices, attention_masks = self.vectorizer.vectorize(input_id)

            return {
                'x_indices': indices,
                'x_attn_mask': attention_masks,
                'x_index': idx,
                'y_humour': target_humour,
                'y_sarcasm': target_sarcasm,
                'y_offensive': target_offensive,
                'y_motivation': target_motivation
            }

        # use image and text
        input_id = self.input_ids[idx] # text
        indices, attention_masks = self.vectorizer.vectorize(input_id)

        img_name = os.path.join(self.root_dir,
                                self.imagelist[idx])
        try:
            image = Image.open(img_name + ".jpg").convert('RGB')
        except:
            print('Except')
            image = Image.open(img_name + ".png").convert('RGB')

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

        return {
            'x_images': preproc_img,
            'x_indices': indices,
            'x_attn_mask': attention_masks,
            'x_index': idx,
            'y_humour': target_humour,
            'y_sarcasm': target_sarcasm,
            'y_offensive': target_offensive,
            'y_motivation': target_motivation
        }