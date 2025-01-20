import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivLoss(nn.Module):
    """KL Divergence for Knowledge Distillation"""
    def __init__(self, temp=4.0):
        super(KLDivLoss, self).__init__()
        self.temp = temp
        
    def forward(self, student_logits, teacher_logits):
        log_student = F.log_softmax(student_logits / self.temp, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temp, dim=1)
        kldiv = F.kl_div(log_student, soft_teacher.detach(), reduction='batchmean')
        return kldiv * (self.temp ** 2)

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining"""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        
        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss

class DistillationLoss(nn.Module):
    """Combined loss for knowledge distillation"""
    def __init__(self, temp=4.0, margin=0.3, lambda_kd=1.0, lambda_tri=1.0):
        super(DistillationLoss, self).__init__()
        self.kl_loss = KLDivLoss(temp=temp)
        self.triplet_loss = TripletLoss(margin=margin)
        self.lambda_kd = lambda_kd
        self.lambda_tri = lambda_tri
        
    def forward(self, student_feats, student_logits, teacher_feats, teacher_logits, targets):
        # Knowledge distillation loss
        kd_loss = self.kl_loss(student_logits, teacher_logits)
        
        # Triplet loss on student features
        tri_loss = self.triplet_loss(student_feats, targets)
        
        # Feature similarity loss
        sim_loss = F.mse_loss(student_feats, teacher_feats.detach())
        
        # Combined loss
        total_loss = self.lambda_kd * kd_loss + self.lambda_tri * tri_loss + sim_loss
        
        return total_loss, {
            'kd_loss': kd_loss.item(),
            'tri_loss': tri_loss.item(),
            'sim_loss': sim_loss.item()
        }
