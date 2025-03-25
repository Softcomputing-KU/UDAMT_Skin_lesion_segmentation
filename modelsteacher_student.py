import copy
import torch.nn as nn

class MeanTeacher(nn.Module):
    def __init__(self, student_model, alpha=0.99):
        super().__init__()
        self.student = student_model
        self.teacher = copy.deepcopy(student_model)
        self.alpha = alpha
        self._freeze_teacher()

    def _freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False

    def update_teacher(self):
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data = self.alpha * t_param.data + (1 - self.alpha) * s_param.data

    def forward(self, x, is_teacher=True):
        return self.teacher(x) if is_teacher else self.student(x)