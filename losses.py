# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Supervised Contrastive Learning을 위한 InfoNCE Loss.
    같은 attack_id를 가진 샘플들은 positive, 다른 샘플들은 negative로 취급합니다.
    """
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): (N, D) 형태의 정규화된 특징 벡터.
            labels (Tensor): (N) 형태의 클래스 레이블 (여기서는 attack_id).
        """
        # 정규화된 특징 벡터 간의 코사인 유사도 계산
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        
        # 자기 자신과의 유사도는 제외하기 위한 마스크 생성
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)
        
        # positive 쌍 마스크 생성: 같은 레이블을 가진 샘플들
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).fill_diagonal_(False).to(features.device)
        
        positive_mask = mask[~torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)].view(labels.shape[0], -1)
        
        # 로짓 계산
        logits = similarity_matrix / self.temperature
        
        # positive 쌍이 없는 경우 loss를 0으로 처리 (에러 방지)
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)
            
        # 손실 계산을 위한 라벨 생성
        # positive 쌍 중 하나를 정답으로 간주
        positive_indices = torch.where(positive_mask)[1]
        
        # 각 행에 대해 positive 쌍이 하나 이상 있다고 가정하고, 첫 번째 positive를 정답으로 사용
        # 이 부분은 구현에 따라 달라질 수 있음 (예: 모든 positive에 대한 loss 평균)
        # 여기서는 간단하게 첫 번째 positive index를 사용
        _, positive_indices = torch.max(positive_mask.float(), dim=1)

        loss = self.criterion(logits, positive_indices)
        return loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    같은 공격 ID(attack_id)를 가진 샘플들을 Positive Pair로, 나머지를 Negative로 취급합니다.
    Ref: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): (N, D) 형태의 정규화된 특징 벡터. (N = batch_size * 2)
            labels (Tensor): (N) 형태의 클래스 레이블 (여기서는 attack_id).
        """
        device = features.device
        
        # 코사인 유사도 행렬 계산
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        
        # 자기 자신과의 비교를 제외하기 위한 마스크
        mask_self = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix.masked_fill(mask_self, -1e9)
        
        # Positive Pair를 찾기 위한 마스크 (같은 레이블을 가진 샘플들)
        labels_matrix = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels_matrix, labels_matrix.T).fill_diagonal_(False)
        
        # positive 쌍이 하나도 없는 행은 손실 계산에서 제외
        positive_per_sample = mask_positive.sum(dim=1)
        valid_samples_mask = positive_per_sample > 0
        
        if not valid_samples_mask.any():
            return torch.tensor(0.0, device=device)
            
        # Log-Sum-Exp 계산 (모든 쌍에 대해)
        exp_similarities = torch.exp(similarity_matrix / self.temperature)
        log_prob = -torch.log(exp_similarities.sum(dim=1, keepdim=True))

        # Positive Pair에 대한 Log-Sum-Exp 계산
        exp_sim_pos = torch.exp((similarity_matrix * mask_positive.float()) / self.temperature)
        # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon) 추가
        sum_exp_sim_pos = exp_sim_pos.sum(dim=1) + 1e-9

        log_prob_pos = torch.log(sum_exp_sim_pos)
        
        # 최종 Loss 계산
        loss = log_prob + log_prob_pos
        loss = loss[valid_samples_mask] # positive가 있는 샘플에 대해서만 평균
        
        # 한 샘플당 positive 쌍의 수로 나누어 정규화
        loss = (loss / positive_per_sample[valid_samples_mask]).mean()
        
        return -loss    

class PatchInfoNCELoss(nn.Module):
    """
    이미지 representation과 patch representation 간의 InfoNCE Loss.
    같은 attack_id를 가진 이미지 representation과 patch representation은 positive,
    다른 attack_id를 가진 이미지 representation과 patch representation은 negative로 취급합니다.
    """
    def __init__(self, temperature=0.1):
        super(PatchInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img_features, patch_features, labels):
        """
        Args:
            img_features (Tensor): (N, D) 형태의 이미지 특징 벡터.
            patch_features (Tensor): (N, D) 형태의 패치 특징 벡터.
            labels (Tensor): (N) 형태의 클래스 레이블 (여기서는 attack_id).
        """
        N = img_features.size(0)

        # 코사인 유사도 계산: 각 이미지 특징과 모든 패치 특징 간의 유사도
        similarity_matrix = F.cosine_similarity(img_features.unsqueeze(1), patch_features.unsqueeze(0), dim=2)

        # positive 쌍 마스크 생성: 같은 attack_id를 가진 이미지와 패치
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).to(img_features.device)

        # negative 쌍 마스크: 다른 attack_id를 가진 이미지와 패치
        negative_mask = ~positive_mask

        # 로짓 계산: 유사도를 temperature로 나누어 scaling
        logits = similarity_matrix / self.temperature

        # InfoNCE Loss 계산을 위한 라벨 생성
        # 각 이미지에 대해, 같은 attack_id를 가진 패치가 정답(positive)
        labels = torch.arange(N).to(img_features.device)  # [0, 1, 2, ..., N-1]

        loss = self.criterion(logits, torch.argmax(positive_mask.int(), dim=1))
        return loss
