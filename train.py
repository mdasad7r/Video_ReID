import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import logging
from tqdm import tqdm

from model.modified_psta import ModifiedPSTA
from model.losses import DistillationLoss
from data.data_loader import get_dataloaders

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

def train_vkd(config):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('train', config.output_dir)
    
    # Create dataloaders
    train_loader_teacher, train_loader_student, query_loader, gallery_loader = get_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
        teacher_seq_len=config.teacher_seq_len,
        student_seq_len=config.student_seq_len
    )
    
    # Create teacher model
    teacher = ModifiedPSTA(
        num_classes=config.num_classes,
        seq_len=config.teacher_seq_len
    ).to(device)
    
    # Load teacher checkpoint
    teacher_ckpt = torch.load(config.teacher_ckpt)
    teacher.load_state_dict(teacher_ckpt['state_dict'])
    teacher.eval()
    
    # Create student model
    student = ModifiedPSTA(
        num_classes=config.num_classes,
        seq_len=config.student_seq_len
    ).to(device)
    
    # Initialize student from teacher except last layer
    student_dict = student.state_dict()
    for k, v in teacher_ckpt['state_dict'].items():
        if 'classifier' not in k:
            student_dict[k] = v
    student.load_state_dict(student_dict)
    
    # Setup training
    criterion = DistillationLoss(
        temp=config.temperature,
        margin=config.margin,
        lambda_kd=config.lambda_kd,
        lambda_tri=config.lambda_tri
    ).to(device)
    
    optimizer = Adam(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)
    
    # Training loop
    best_mAP = 0
    for epoch in range(config.num_epochs):
        student.train()
        
        for batch_idx, ((teacher_imgs, pids, _), (student_imgs, _, _)) in enumerate(
            zip(train_loader_teacher, train_loader_student)):
            
            teacher_imgs = teacher_imgs.to(device)
            student_imgs = student_imgs.to(device)
            pids = pids.to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                t_feats, t_logits = teacher(teacher_imgs, return_logits=True)
            
            # Get student predictions
            s_feats, s_logits = student(student_imgs, return_logits=True)
            
            # Calculate loss
            loss, loss_dict = criterion(s_feats, s_logits, t_feats, t_logits, pids)
            
            # Update student
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                logger.info(f'Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.4f}')
                for k, v in loss_dict.items():
                    logger.info(f'{k}: {v:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config.save_period == 0:
            state = {
                'state_dict': student.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(config.output_dir, f'epoch_{epoch}.pth'))
            
        # Evaluate
        if (epoch + 1) % config.eval_period == 0:
            mAP = evaluate(student, query_loader, gallery_loader, device)
            logger.info(f'Epoch: {epoch} mAP: {mAP:.4f}')
            
            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(state, os.path.join(config.output_dir, 'best.pth'))
                
    logger.info(f'Best mAP: {best_mAP:.4f}')

def evaluate(model, query_loader, gallery_loader, device):
    model.eval()
    
    # Extract features
    query_feats = []
    query_pids = []
    gallery_feats = []
    gallery_pids = []
    
    with torch.no_grad():
        for imgs, pids, _ in tqdm(query_loader):
            imgs = imgs.to(device)
            feats = model(imgs)
            query_feats.append(feats.cpu())
            query_pids.extend(pids.numpy())
            
        for imgs, pids, _ in tqdm(gallery_loader):
            imgs = imgs.to(device)
            feats = model(imgs)
            gallery_feats.append(feats.cpu())
            gallery_pids.extend(pids.numpy())
    
    query_feats = torch.cat(query_feats, dim=0)
    gallery_feats = torch.cat(gallery_feats, dim=0)
    
    # Calculate distances
    m, n = query_feats.size(0), gallery_feats.size(0)
    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(query_feats, gallery_feats.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()
    
    # Compute mAP
    from utils.metrics import mean_ap
    mAP = mean_ap(distmat, query_pids, gallery_pids, query_camids, gallery_camids)
    
    return mAP

if __name__ == '__main__':
    from config import get_default_config
    config = get_default_config()
    train_vkd(config)
