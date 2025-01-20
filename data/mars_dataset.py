import os.path as osp
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import glob
import re
import os
import numpy as np
from PIL import Image

class MarsDataset(Dataset):
    def __init__(self, root, subset='train', seq_len=8, sample='random', transform=None):
        self.root = root
        self.subset = subset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        
        self.prepare_data()
        
    def prepare_data(self):
        if self.subset == 'train':
            self.data_path = osp.join(self.root, 'bbox_train')
        else:
            self.data_path = osp.join(self.root, 'bbox_test')
            
        self.videos = []
        self.pids = []
        self.cams = []
        
        pattern = re.compile(r'C(\d+)/(\d+)/')
        for video_path in glob.glob(osp.join(self.data_path, '*/*/*.jpg')):
            pid, cam = map(int, pattern.search(video_path).groups())
            if self.subset == 'train' or (self.subset != 'train' and pid != -1):
                self.videos.append(video_path)
                self.pids.append(pid)
                self.cams.append(cam)
                
        self.num_videos = len(self.videos)
        
    def __len__(self):
        return self.num_videos
        
    def __getitem__(self, index):
        video_path = self.videos[index]
        pid = self.pids[index]
        cam = self.cams[index]
        
        if self.sample == 'random':
            frame_indices = self.random_sample(len(glob.glob(osp.join(video_path, '*.jpg'))))
        else:
            frame_indices = self.dense_sample(len(glob.glob(osp.join(video_path, '*.jpg'))))
            
        imgs = []
        for idx in frame_indices:
            img_path = osp.join(video_path, f'F{idx:04d}.jpg')
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
            
        imgs = torch.stack(imgs, 0)
        return imgs, pid, cam
        
    def random_sample(self, num_frames):
        if num_frames < self.seq_len:
            indices = np.arange(num_frames)
            indices = np.concatenate([indices, np.random.choice(indices, self.seq_len - num_frames)])
        else:
            indices = sorted(np.random.choice(num_frames, self.seq_len, replace=False))
        return indices
        
    def dense_sample(self, num_frames):
        if num_frames < self.seq_len:
            indices = np.arange(num_frames)
            indices = np.concatenate([indices, np.random.choice(indices, self.seq_len - num_frames)])
        else:
            stride = num_frames / self.seq_len
            indices = np.array([int(i * stride) for i in range(self.seq_len)])
        return indices

class RandomIdentitySampler:
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances
        
        pid_idx = {}
        for idx, pid in enumerate(data_source.pids):
            if pid not in pid_idx:
                pid_idx[pid] = []
            pid_idx[pid].append(idx)
            
        self.pids = list(pid_idx.keys())
        self.pid_idx = pid_idx
        
    def __len__(self):
        return self.batch_size * len(self.pids) // self.num_pids_per_batch
        
    def __iter__(self):
        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = self.pid_idx[pid]
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid] = batch_idxs
                    batch_idxs = []
                    
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_ret = []
        
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs_ret.extend(batch_idxs_dict[pid])
                avai_pids.remove(pid)
                
        return iter(batch_idxs_ret)
