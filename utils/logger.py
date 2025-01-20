# utils/logger.py

import logging
import os
import sys
import time
from datetime import datetime

class Logger:
    """Custom logger for training progress."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging to both file and console
        time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_file = os.path.join(save_dir, f'log_{time_str}.txt')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize training statistics
        self.epoch_stats = {}
        self.running_stats = {}
        self.best_performance = {'mAP': 0.0, 'rank1': 0.0}
        
    def log_training(self, epoch, batch_idx, num_batches, losses, lr, speed=None):
        """Log training progress."""
        msg = f'Epoch: [{epoch}][{batch_idx}/{num_batches}]\t'
        
        # Log learning rate
        msg += f'lr: {lr:.2e}\t'
        
        # Log all losses
        for k, v in losses.items():
            msg += f'{k}: {v:.4f}\t'
            
        # Log training speed
        if speed is not None:
            msg += f'Speed: {speed:.1f} samples/s'
            
        self.logger.info(msg)
        
    def log_eval(self, epoch, metrics, is_best=False):
        """Log evaluation results."""
        msg = f'Evaluation at epoch: {epoch}\t'
        
        for k, v in metrics.items():
            msg += f'{k}: {v:.2%}\t'
            
        if is_best:
            msg += ' (Best)'
            
        self.logger.info(msg)
        
    def update_best_performance(self, metrics):
        """Update best performance metrics."""
        updated = False
        
        if metrics['mAP'] > self.best_performance['mAP']:
            self.best_performance['mAP'] = metrics['mAP']
            updated = True
            
        if metrics['rank1'] > self.best_performance['rank1']:
            self.best_performance['rank1'] = metrics['rank1']
            updated = True
            
        return updated
        
    def log_epoch_statistics(self, epoch, epoch_time):
        """Log statistics for entire epoch."""
        msg = f'Epoch {epoch} completed in {epoch_time:.2f}s\t'
        
        # Log epoch averages
        for k, v in self.epoch_stats.items():
            msg += f'Avg {k}: {v:.4f}\t'
            
        self.logger.info(msg)
        self.epoch_stats = {}  # Reset for next epoch
        
    def start_epoch(self):
        """Start timing an epoch."""
        self.epoch_start_time = time.time()
        
    def end_epoch(self, epoch):
        """End timing an epoch and log statistics."""
        epoch_time = time.time() - self.epoch_start_time
        self.log_epoch_statistics(epoch, epoch_time)
        
    def log_hyperparameters(self, config):
        """Log training hyperparameters."""
        self.logger.info('Training hyperparameters:')
        for k, v in vars(config).items():
            self.logger.info(f'{k}: {v}')
            
    def log_architecture(self, model):
        """Log model architecture."""
        self.logger.info('Model architecture:')
        self.logger.info(str(model))
        
    def save_model(self, state, is_best=False):
        """Save model checkpoint."""
        epoch = state['epoch']
        fpath = os.path.join(self.save_dir, f'checkpoint_ep{epoch}.pth')
        torch.save(state, fpath)
        
        if is_best:
            best_fpath = os.path.join(self.save_dir, 'model_best.pth')
            torch.save(state, best_fpath)
