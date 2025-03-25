import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class ISIC2018Dataset(Dataset):
    def __init__(self, root_dir, split='train', labeled_ratio=0.05):
        self.root_dir = root_dir
        self.split = split
        self.labeled_ratio = labeled_ratio
        self.image_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(os.path.join(root_dir, 'masks')) if f.endswith('.png')])
        
        # Split labeled and unlabeled data
        num_labeled = int(len(self.image_files) * labeled_ratio)
        if split == 'train':
            self.image_files = self.image_files[:num_labeled]  # Labeled
            self.mask_files = self.mask_files[:num_labeled]
        elif split == 'unlabeled':
            self.image_files = self.image_files[num_labeled:]  # Unlabeled
            self.mask_files = self.mask_files[num_labeled:]
        
        self.transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.image_files[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.mask_files[idx])
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image = self.transform(image)
        mask = Resize((224, 224))(mask)
        mask = ToTensor()(mask).squeeze(0)
        return image, mask