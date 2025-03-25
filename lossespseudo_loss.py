import torch
import torch.nn as nn

class PseudoLabelLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, student_pred, teacher_pred, uncertainty_map):
        mask = (uncertainty_map < self.threshold).float()
        loss = self.bce_loss(student_pred, teacher_pred.sigmoid().detach())
        masked_loss = (loss * mask).mean()
        return masked_loss