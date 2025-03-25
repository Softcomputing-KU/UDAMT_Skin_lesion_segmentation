import torch
from torch.utils.data import DataLoader
from data.isic2018 import ISIC2018Dataset
from models.dual_head_unet import DualHeadSeg
from models.teacher_student import MeanTeacher
from losses.supervised_loss import SupervisedLoss
from losses.pseudo_loss import PseudoLabelLoss
from utils.augmentations import WeakStrongAugmentation
import yaml

# Load config
with open("configs/udamt.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize datasets
labeled_dataset = ISIC2018Dataset(root_dir="data/ISIC2018", split='train', labeled_ratio=config['labeled_ratio'])
unlabeled_dataset = ISIC2018Dataset(root_dir="data/ISIC2018", split='unlabeled')
labeled_loader = DataLoader(labeled_dataset, batch_size=config['batch_size'], shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config['batch_size'], shuffle=True)

# Initialize models
student = DualHeadSeg()
teacher = MeanTeacher(student, alpha=config['ema_alpha'])
optimizer = torch.optim.Adam(student.parameters(), lr=config['lr'])

# Loss functions
supervised_loss = SupervisedLoss()
pseudo_loss = PseudoLabelLoss(threshold=config['uncertainty_threshold'])

# Training loop
for epoch in range(config['total_epochs']):
    # Supervised phase (first 20 epochs)
    if epoch < 20:
        for images, masks in labeled_loader:
            main_pred, aux_pred = student(images)
            loss = supervised_loss(main_pred, masks) + supervised_loss(aux_pred, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Semi-supervised phase
    else:
        for (labeled_imgs, labeled_masks), unlabeled_imgs in zip(labeled_loader, unlabeled_loader):
            # Labeled loss
            main_pred, aux_pred = student(labeled_imgs)
            sup_loss = supervised_loss(main_pred, labeled_masks) + supervised_loss(aux_pred, labeled_masks)
            
            # Generate pseudo-labels
            with torch.no_grad():
                teacher_pred, _ = teacher(unlabeled_imgs, is_teacher=True)
                uncertainty = teacher.calculate_uncertainty(unlabeled_imgs)
            
            # Pseudo-label loss
            student_pred, _ = student(unlabeled_imgs)
            unsup_loss = pseudo_loss(student_pred, teacher_pred, uncertainty)
            
            # Total loss
            total_loss = sup_loss + config['lambda_pseudo'] * unsup_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            teacher.update_teacher()