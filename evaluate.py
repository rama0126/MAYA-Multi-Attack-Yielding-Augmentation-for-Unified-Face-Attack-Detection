import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# 로컬 모듈 임포트
from dataset import FASValidationDataset
from model import ClipFas

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
# evaluate.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
from dataset import FASDataset
from model import ClipFas
import logging
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------
# Metrics 유틸리티
# ------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

def load_predictions(pred_txt):
    """
    preds_txt: 
      Data-val/05389.png 0.00616
      Data-val/05390.png 0.03712
      ...
    """
    files, scores = [], []
    with open(pred_txt, 'r') as f:
        for line in f:
            fn, sc = line.strip().split()
            files.append(fn)
            scores.append(float(sc))
    return np.array(files), np.array(scores)

def load_labels(label_txt):
    """
    labels_txt:
      Data-val/00012.png 0 1_0_0
      Data-val/00013.png 0 1_0_0
      ...
    공격 코드(3자리 중 첫째)가 0이면 bona-fide, >0이면 attack 으로 본다.
    """
    files, y = [], []
    with open(label_txt, 'r') as f:
        for line in f:
            fn, label, atk = line.strip().split()
            label = int(label)
            files.append(fn)
            y.append(label)
    return np.array(files), np.array(y)

def align_preds_labels(pred_files, preds, label_files, labels):
    # 두 리스트의 교집합만 남기고, 이름순 정렬
    common = np.intersect1d(pred_files, label_files)
    common.sort()
    idx_p = np.argsort(pred_files)
    idx_l = np.argsort(label_files)
    # pick only common
    pf_sorted, pr_sorted = pred_files[idx_p], preds[idx_p]
    lf_sorted, lb_sorted = label_files[idx_l], labels[idx_l]
    mask_p = np.in1d(pf_sorted, common)
    mask_l = np.in1d(lf_sorted, common)
    return pf_sorted[mask_p], pr_sorted[mask_p], lb_sorted[mask_l]

def compute_rates(y_true, y_pred):
    # attack=1, bona-fide=0
    N_attack = np.sum(y_true==1)
    N_live   = np.sum(y_true==0)
    FN = np.sum((y_true==1) & (y_pred==0))
    FP = np.sum((y_true==0) & (y_pred==1))
    apcer = FN / N_attack
    bpcer = FP / N_live
    acer  = (apcer + bpcer) / 2
    return acer, apcer, bpcer

def calculate_metrics(y_true, y_score, threshold=0.5, 
                      AUC=True):
    """
    한 임계치에서의 최종 지표 계산
    """
    y_pred = (y_score >= threshold).astype(int)
    acer, apcer, bpcer = compute_rates(y_true, y_pred)
    if AUC:
        auc = roc_auc_score(y_true, y_score)
    else:
        fpr, tpr = None, None
        auc = None
    acc  = accuracy_score(y_true, y_pred)
    return {
        'threshold': threshold,
        'APCER': apcer,
        'BPCER': bpcer,
        'ACER': acer,
        'ACC': acc,
        'AUC': auc
    }

def find_optimal_threshold(y_true, y_score):
    """
    validation 세트에서 ACER를 최소화하는 최적의 threshold 탐색
    """
    thresholds = np.linspace(0, 1, 1000001)
    best = None
    best_acer = 1.0
    for t in thresholds:
        acer, _, _ = compute_rates(y_true, (y_score>=t).astype(int))
        if acer < best_acer:
            best_acer = acer
            best = t
    return best, best_acer

def threshold_at_max_fpr(y_true, y_score, max_fpr=0.1):
    """
    validation ROC 곡선에서 FPR<=max_fpr 조건을 만족하는
    가장 높은 threshold 반환
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.where(fpr <= max_fpr)[0]
    if len(idx)==0:
        return None
    return thr[idx].max()

# ------------------------------------------------------------------
# 실제 검증 루프
# ------------------------------------------------------------------
from tqdm import tqdm
import torch.nn.functional as F
def evaluate_model(model, dataloader, device, attack_int_to_str = None):
    model.eval()
    all_probs = []
    all_labels = []
    all_atks = []
    # attack_int_to_str는 dataset에 정의되어 있지 않으면 None으로 설정
    # dataloader.dataset이 attack_int_to_str을 가지고 있다면 사용, 아니면 None
    # attack_int_to_str은 공격 유형을 정수에서 문자열로 매핑하는 딕셔너리
    # 만약 dataloader.dataset이 attack_int_to_str을 가지고 있지 않다면,
    attack_int_to_str = dataloader.dataset.attack_int_to_str if hasattr(dataloader.dataset, 'attack_int_to_str') else attack_int_to_str

    tmp_txt_file = './tmp_predict.txt'
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Predicting")
        # dataset.py가 (image, label)을 반환하므로, label은 무시
        for images, labels, attack_types in pbar:
            images = images.to(device)
            
            # 모델 포워드 패스
            outputs = model(images)
            
            # Softmax를 적용하여 확률 계산
            # outputs shape: [batch_size, num_classes] (num_classes=2)
            # 클래스 0: Fake, 클래스 1: Live
            probabilities = F.softmax(outputs, dim=1)
            
            # Live(클래스 1)일 확률을 스코어로 사용
            live_scores = probabilities[:, 1].cpu().numpy()
            labels = labels.cpu().numpy()  # 실제 레이블
            attack_types = attack_types.cpu().numpy() # 공격 유형
            # 현재 배치의 결과 저장 (경로 정보는 dataloader에서 직접 가져오기 어려우므로,
            # dataset에서 순서대로 가져와야 함. dataloader의 shuffle=False가 중요)
            all_probs.extend(live_scores)
            all_labels.extend(labels)
            all_atks.extend(attack_types)
    
    # 최종 결과를 numpy 배열로 변환
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_atks = np.array(all_atks)
    
    
    # ACER 계산을 위한 최적의 threshold 찾기
    optimal_threshold, optimal_acer = find_optimal_threshold(all_labels, all_probs)
    logging.info(f"Optimal Threshold: {optimal_threshold:.4f}, Optimal ACER: {optimal_acer:.4f}")
    # 최적의 threshold에서의 지표 계산
    metrics = calculate_metrics(all_labels, all_probs, threshold=optimal_threshold)
    
    results = {
        'overall': metrics,
        'by_attack_type': {}
    }
    # 공격 유형별로 지표 계산
    unique_attack_types = np.unique(all_atks)
    live_mask = all_labels == 1
    for attack_type in unique_attack_types:
        attack_mask = all_atks == attack_type
        if np.sum(attack_mask) == 0:
            continue  # 해당 공격 유형에 대한 샘플이 없는 경우 건너뜀
        # mask = attack_mask U live_mask
        mask = attack_mask
        attack_labels = all_labels[mask]
        attack_probs = all_probs[mask]
        attack_metrics = calculate_metrics(attack_labels, attack_probs, threshold=optimal_threshold)
        attack_type_name = attack_int_to_str[attack_type] if attack_int_to_str else f'ATTACK_{attack_type}'
        results['by_attack_type'][attack_type_name] = attack_metrics
        results['by_attack_type'][attack_type_name]['Num_Samples'] = np.sum(attack_mask)
    # 전체 결과에 대한 ACER, ACC, AUC 추가
    results['overall']['Optimal_Threshold'] = optimal_threshold
    results['overall']['Optimal_ACER'] = optimal_acer
    results['overall']['Num_Samples'] = len(all_labels)
    return results
        
        
        


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    val_dataset = FASValidationDataset(
        base_dir='/workspace/datasets/FAS2/Data-val_preprocessed/images',
        data_list_path='/workspace/datasets/FAS2/Data-val_preprocessed/val_list.txt'
    )
    model_path = '/workspace/FAS_ICCV2/checkpoints/adapter/clip_fas_iter_8000.pth'
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        num_workers=1,
        pin_memory=False,
        shuffle=False
    )
    model = ClipFas(
        num_classes=2, 
        model_name='ViT-L-14', 
        pretrained='laion2b_s32b_b82k',
        finetune_strategy='adapter'  # 학습 시 사용한 전략과 동일해야 함
    )
    
    # DataParallel로 학습했다면, state_dict의 키 이름이 다를 수 있음
    state_dict = torch.load(model_path, map_location=device)
    
    # DataParallel 래퍼가 추가한 'module.' 접두사 처리
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    is_data_parallel = any(key.startswith('module.') for key in state_dict.keys())
    
    if is_data_parallel:
        for k, v in state_dict.items():
            name = k[7:] # 'module.' 제거
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model = model.to(device)
    
    results = evaluate_model(model, val_loader, device)

    overall_metrics = results['overall']
    logging.info(f"Overall Metrics: ACER={overall_metrics['ACER']:.4f}, ACC={overall_metrics['ACC']:.4f}, AUC={overall_metrics['AUC']:.4f}")

    for attack_type, metrics in results['by_attack_type'].items():
        attack_type_name = 'LIVE' if attack_type == 0 else f'ATTACK_{attack_type}'
        logging.info(f"Attack Type: {attack_type_name} - ACER={metrics['ACER']:.4f}, ACC={metrics['ACC']:.4f}, AUC={metrics['AUC']:.4f}")

if __name__ == '__main__':
    main()