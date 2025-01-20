from yacs.config import CfgNode as CN

def get_default_config():
    cfg = CN()
    
    # Model
    cfg.model = CN()
    cfg.model.name = 'modified_psta'
    cfg.model.num_classes = 625  # MARS dataset
    cfg.model.teacher_seq_len = 8
    cfg.model.student_seq_len = 2
    
    # Data
    cfg.data = CN()
    cfg.data.root = 'data/mars'
    cfg.data.height = 256
    cfg.data.width = 128
    cfg.data.batch_size = 32
    cfg.data.num_instances = 4
    cfg.data.num_workers = 4
    
    # Training
    cfg.train = CN()
    cfg.train.num_epochs = 400
    cfg.train.lr = 3.5e-4
    cfg.train.weight_decay = 5e-4
    cfg.train.milestones = [150, 225]
    cfg.train.temperature = 4.0
    cfg.train.margin = 0.3
    cfg.train.lambda_kd = 1.0
    cfg.train.lambda_tri = 1.0
    
    # Paths
    cfg.teacher_ckpt = '/content/drive/MyDrive/Colab_Checkpoints_for_PI/Mars_rank1_0.915.pth'
    cfg.output_dir = 'checkpoints/mars_vkd'
    
    # Logging
    cfg.save_period = 50
    cfg.eval_period = 50
    
    return cfg
