import torch
import torch.nn as nn
import torch.nn.functional as F
class SingleCenterLoss(nn.Module):
    """
    Single Center Loss

    Reference:
    J Li, Frequency-aware Discriminative Feature Learning Supervised by Single-CenterLoss for Face Forgery Detection, CVPR 2021.

    Parameters:
        m (float): margin parameter. 
        D (int): feature dimension.
        C (vector): learnable center.
    """
    def __init__(self, m = 0.3, D = 1000, use_gpu=True):
        super(SingleCenterLoss, self).__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.use_gpu = use_gpu
        self.l2loss = nn.MSELoss(reduction = 'none')
        if self.use_gpu:
            self.C = nn.Parameter(torch.randn(self.D).cuda())
        else:
            self.C = nn.Parameter(torch.randn(self.D))
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        eud_mat = torch.sqrt(self.l2loss(x, self.C.expand(batch_size, self.C.size(0))).sum(dim=1, keepdim=True))
        labels = labels.unsqueeze(1)
        fake_count = labels.sum()
        real_count = batch_size - fake_count
        dist_fake = (eud_mat * labels.float()).clamp(min=1e-12, max=1e+12).sum()     # fake = 1
        dist_real = (eud_mat * (1 - labels.float())).clamp(min=1e-12, max=1e+12).sum()  # live = 0
        if real_count != 0:
            dist_real /= real_count
        if fake_count != 0:
            dist_fake /= fake_count
        max_margin = dist_real - dist_fake + self.margin  # Changed order to make real samples spread out
        if max_margin < 0:
            max_margin = 0
        loss = dist_real + max_margin  # Changed to minimize distance for fake samples
        return loss
    
class ImprovedSingleCenterLoss(nn.Module):
    def __init__(self, m=0.3, D=1000, use_gpu=True, hard_weight=1.0):
        super().__init__()
        self.m = m
        self.D = D
        self.margin = self.m * torch.sqrt(torch.tensor(self.D).float())
        self.use_gpu = use_gpu
        self.hard_weight = hard_weight
        self.eps = 1e-6

        self.C = nn.Parameter(torch.randn(D).cuda() if use_gpu else torch.randn(D))
        self.l2loss = nn.MSELoss(reduction='none')

    def forward(self, x, labels):
        batch_size = x.size(0)
        center = self.C.expand(batch_size, -1)

        # Calculate Euclidean distance
        dist = torch.sqrt(self.l2loss(x, self.C.expand(batch_size, self.C.size(0))).sum(dim=1, keepdim=True))

        labels = labels.float()
        fake_mask = labels == 0
        real_mask = labels == 1

        # Calculate mean distances
        fake_dist = dist[fake_mask].mean() if fake_mask.any() else torch.tensor(0.0).to(dist.device)
        real_dist = dist[real_mask].mean() if real_mask.any() else torch.tensor(0.0).to(dist.device)

        # Adaptive margin
        adaptive_margin = real_dist.detach() - fake_dist.detach()
        margin_term = F.relu(adaptive_margin + self.margin)

        # Hard real sample repulsion
        real_hard = dist[real_mask][dist[real_mask] < real_dist]
        repulsion_loss = real_hard.mean() if len(real_hard) > 0 else torch.tensor(0.0).to(dist.device)

        # Final loss
        loss = fake_dist + margin_term + self.hard_weight * repulsion_loss
        return loss
    
class ConcentricRingLoss(nn.Module):
    def __init__(self, D, angular_margin=0.5, euclidean_margin=10.0, use_gpu=True):
        super(ConcentricRingLoss, self).__init__()
        self.angular_margin = angular_margin  # β in angular margin loss
        self.euclidean_margin = euclidean_margin  # M in Euclidean margin loss
        self.use_gpu = use_gpu
        self.D = D  # feature dimension
        
        # Learnable parameters
        self.W = nn.Parameter(torch.randn(1, D).cuda() if use_gpu else torch.randn(1, D))  # Single output for binary
        self.r1 = nn.Parameter(torch.tensor(1.0).cuda() if use_gpu else torch.tensor(1.0))  # target norm for real
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Normalize weights and features for angular calculation
        W_norm = F.normalize(self.W, dim=1)  # [1, D]
        x_orig_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # Store original norm
        x_norm = F.normalize(x, dim=1)  # [B, D]
        
        # Calculate cosine similarities
        cos_theta = torch.mm(x_norm, W_norm.t())  # [B, 1]
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)
        
        # Add angular margin β for real samples
        theta_with_margin = theta.clone()
        real_mask = (labels == 1)  # [B, 1]
        theta_with_margin[real_mask] += self.angular_margin  # Add margin for real class
        logits = x_orig_norm * torch.cos(theta_with_margin)  # Re-scale by original magnitude
        
        # Angular margin loss
        L_ang = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Euclidean margin loss
        real_dist = (x_orig_norm[labels == 1] - self.r1).pow(2).mean() if (labels == 1).any() else torch.tensor(0.0).to(x.device)
        fake_dist = (x_orig_norm[labels == 0] - (self.r1 + self.euclidean_margin)).pow(2).mean() if (labels == 0).any() else torch.tensor(0.0).to(x.device)
        L_euc = real_dist + fake_dist
        
        # Total loss
        loss = L_ang + L_euc
        
        return loss