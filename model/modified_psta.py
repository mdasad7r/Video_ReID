import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.resnet import ResNet
from model.SRA import SRA

class ModifiedSTAM(nn.Module):
    """Modified STAM module without TRA"""
    def __init__(self, inplanes, mid_planes, num):
        super(ModifiedSTAM, self).__init__()
        self.num = num
        
        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(inplanes, mid_planes, 1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention only
        self.sra = SRA(inplanes=inplanes, num=num)
        
        # Feature refinement
        self.conv_block = nn.Sequential(
            nn.Conv2d(inplanes, mid_planes, 1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, mid_planes, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_planes, inplanes, 1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        b, t, c, h, w = x.size()
        x_flat = x.view(b * t, c, h, w)
        
        # Generate embeddings
        feat_vect = self.avg_pool(x_flat).view(b, t, -1)
        embed_feat = self.embedding(x_flat).view(b, t, -1, h, w)
        
        # Apply spatial attention
        out = self.sra(x, x_flat, embed_feat, feat_vect)
        out = self.conv_block(out)
        out = out.view(b, -1, c, h, w)
        
        return out

class ModifiedPSTA(nn.Module):
    """Modified PSTA without temporal attention"""
    def __init__(self, num_classes, model_name='resnet50', pretrain_choice='imagenet', seq_len=2):
        super(ModifiedPSTA, self).__init__()
        
        self.in_planes = 2048
        self.base = ResNet()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.planes = 1024
        self.mid_channel = 256

        # Feature reduction
        self.down_channel = nn.Sequential(
            nn.Conv2d(self.in_planes, self.planes, 1, bias=False),
            nn.BatchNorm2d(self.planes),
            nn.ReLU(inplace=True)
        )
        
        # Pyramid stages
        self.layer1 = ModifiedSTAM(self.planes, self.mid_channel, '1')
        self.layer2 = ModifiedSTAM(self.planes, self.mid_channel, '2')
        self.layer3 = ModifiedSTAM(self.planes, self.mid_channel, '3')
        
        # Classification layers
        self.bottleneck = nn.ModuleList([
            nn.BatchNorm1d(self.planes) for _ in range(3)
        ])
        self.classifier = nn.ModuleList([
            nn.Linear(self.planes, num_classes) for _ in range(3)
        ])
        
        # Initialize weights
        for b in self.bottleneck:
            b.bias.requires_grad_(False)
            nn.init.constant_(b.weight, 1)
            nn.init.constant_(b.bias, 0)
            
        for c in self.classifier:
            nn.init.normal_(c.weight, std=0.001)
            
    def forward(self, x, return_logits=False):
        b, t, c, h, w = x.size()
        
        # Extract base features
        x = x.view(b * t, c, h, w)
        feat_map = self.base(x)
        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, feat_map.size(2), feat_map.size(3))
        
        # Progressive refinement
        features = []
        stage_features = []
        
        # Stage 1
        feat1 = self.layer1(feat_map)
        feat1_pool = torch.mean(feat1, 1)
        feat1_pool = F.adaptive_avg_pool2d(feat1_pool, (1, 1)).view(b, -1)
        features.append(feat1_pool)
        stage_features.append(feat1_pool)
        
        # Stage 2
        feat2 = self.layer2(feat1)
        feat2_pool = torch.mean(feat2, 1)
        feat2_pool = F.adaptive_avg_pool2d(feat2_pool, (1, 1)).view(b, -1)
        stage_features.append(feat2_pool)
        feat2_final = torch.stack(stage_features, 1)
        feat2_final = torch.mean(feat2_final, 1)
        features.append(feat2_final)
        
        # Stage 3
        feat3 = self.layer3(feat2)
        feat3_pool = torch.mean(feat3, 1)
        feat3_pool = F.adaptive_avg_pool2d(feat3_pool, (1, 1)).view(b, -1)
        stage_features.append(feat3_pool)
        feat3_final = torch.stack(stage_features, 1)
        feat3_final = torch.mean(feat3_final, 1)
        features.append(feat3_final)
        
        # Apply bottleneck
        bn_features = []
        for i, f in enumerate(features):
            bn_features.append(self.bottleneck[i](f))
            
        if self.training:
            # Get classification scores
            cls_scores = []
            for i, f in enumerate(bn_features):
                cls_scores.append(self.classifier[i](f))
                
            if return_logits:
                return bn_features[-1], cls_scores[-1]
            return bn_features[-1]
        else:
            return bn_features[-1]
