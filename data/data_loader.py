import torchvision.transforms as T
from data.mars_dataset import MarsDataset, RandomIdentitySampler
from torch.utils.data import DataLoader

def get_transform(is_train=True):
    """Get data transformation pipeline"""
    if is_train:
        transform = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5)
        ])
    else:
        transform = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    return transform

def get_dataloaders(data_root, batch_size=32, num_instances=4, 
                   num_workers=4, teacher_seq_len=8, student_seq_len=2):
    """Get dataloaders for training and testing"""
    
    # Training dataloaders
    train_set_teacher = MarsDataset(
        root=data_root,
        subset='train',
        seq_len=teacher_seq_len,
        sample='random',
        transform=get_transform(True)
    )
    
    train_set_student = MarsDataset(
        root=data_root,
        subset='train',
        seq_len=student_seq_len,
        sample='random',
        transform=get_transform(True)
    )
    
    train_loader_teacher = DataLoader(
        train_set_teacher,
        sampler=RandomIdentitySampler(train_set_teacher, batch_size, num_instances),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    train_loader_student = DataLoader(
        train_set_student,
        sampler=RandomIdentitySampler(train_set_student, batch_size, num_instances),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Test dataloaders
    query_set = MarsDataset(
        root=data_root,
        subset='query',
        seq_len=student_seq_len,
        sample='dense',
        transform=get_transform(False)
    )
    
    gallery_set = MarsDataset(
        root=data_root,
        subset='gallery',
        seq_len=student_seq_len,
        sample='dense',
        transform=get_transform(False)
    )
    
    query_loader = DataLoader(
        query_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader_teacher, train_loader_student, query_loader, gallery_loader
