import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class CustomDataset(Dataset):
    def __init__(self, image_paths, captions, tokenizer, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']

        text_inputs = self.tokenizer(caption, padding="max_length", truncation=True)        
        target = {
            "input_ids": text_inputs['input_ids'],
            "attention_mask": text_inputs['attention_mask']
        }
        
        return image, target