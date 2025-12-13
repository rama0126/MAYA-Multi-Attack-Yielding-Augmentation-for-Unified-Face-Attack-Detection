# train.py (Iteration-based saving)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import itertools # 무한 반복 데이터로더를 위해 추가

# 로컬 모듈 임포트
from dataset import FASDatasetWithCLIP
from sampler import AttackTypeBatchSampler
from model import ClipFas
from sam import SAM
import random
import numpy as np
from dataset import FASSimCLRDataset    # 수정된 데이터셋
from sampler import FinalContrastiveSampler
from losses import SupervisedContrastiveLoss, PatchInfoNCELoss

from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict
# 로깅 설정
import gc
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    

# 재현성을 위한 seed 고정
def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용 시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")

def main(args):
    # --- 하이퍼파라미터 및 설정 ---
    # train_base_dir = '/workspace/datasets/FAS2/Data-train_preprocessed'
    # train_list_path = '/workspace/datasets/FAS2/Data-train_preprocessed/train_label.txt'
    train_base_dir = args.train_base_dir
    train_list_path = os.path.join(train_base_dir, 'train_label.txt')
    set_seed(42)  # 재현성을 위한 시드 설정
    logging.info("Starting training...")
    # --- SOLUTION: Iteration-based training parameters ---
    max_iterations = 20000  # 총 학습할 이터레이션 수
    save_interval = 500   # 2000 이터레이션마다 모델 저장
    model_save_dir = './checkpoints' # 모델 저장 디렉토리
    os.makedirs(model_save_dir, exist_ok=True)
    finetune_strategy = 'Swin' # 'adapter', 'partial', 'linear_probe', 'full' 중 선택

    batch_size = 16*8 # 배치 크기 (9개의 공격 유형 * 10개 샘플)
    image_size = 224
    learning_rate = 2e-4
    weight_decay = 1e-3
    K_CLASSES = 9  # 한 배치에 Live 1개 + Fake 8개
    P_SAMPLES = 4  # 클래스당 8개 샘플

    sam_rho = 0.05
    use_sam = False
    lambda_contrastive = 0.05 # Contrastive Loss 가중치

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- 데이터로더 설정 ---
    logging.info("Setting up data loaders...")
    # --- 학습 데이터로더 설정 ---
    from dataset import FASSimCLRDataset
    train_dataset = FASSimCLRDataset(base_dir=train_base_dir, data_list_path=train_list_path, )
    train_batch_sampler = FinalContrastiveSampler(dataset=train_dataset,batch_k=K_CLASSES, batch_p=P_SAMPLES)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,  # CPU 메모리 부족 문제로 num_workers를 0으로 설정
        pin_memory=False
    )
    # 무한 반복 데이터로더 생성
    train_loader_iter = itertools.cycle(train_loader)

    # --- 모델, 손실 함수, 옵티마이저 설정 ---
    logging.info("Initializing model, criterion, and optimizer...")
    model = ClipFas(
        num_classes=2, 
        model_name='ViT-L-14', 
        pretrained='laion2b_s32b_b82k',
        finetune_strategy=finetune_strategy
    )
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    

    criterion = nn.CrossEntropyLoss()
    criterion_contrastive = PatchInfoNCELoss(temperature=0.1)  # Contrastive Loss 정의
    base_optimizer = optim.AdamW

    if use_sam:
    # SAM 옵티마이저 설정
        logging.info("Using SAM optimizer.")
        optimizer = SAM(model.parameters(), base_optimizer, rho=sam_rho, lr=learning_rate, weight_decay=weight_decay)
    else:
        logging.info("Using standard AdamW optimizer.")
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max_iterations*1.1), eta_min=1e-6)
    
    # --- 학습 루프 (Iteration 기반) ---
    logging.info(f"Starting training for {max_iterations} iterations...")
    model.train()
    scaler = GradScaler()

    pbar = tqdm(range(1, max_iterations + 1), desc="Training Iterations")
    first_iter = True  # 첫 번째 이터레이션 여부
    for current_iter in pbar:
        # 데이터로더에서 다음 배치 가져오기
        try:
            view1, view2, live_fake_labels, attack_ids = next(train_loader_iter)
        except StopIteration:
            # 이론적으로 itertools.cycle 때문에 발생하지 않음
            train_loader_iter = itertools.cycle(train_loader)
            view1, view2, live_fake_labels, attack_ids = next(train_loader_iter)

        view1, view2 = view1.to(device), view2.to(device)
        attack_ids = attack_ids.to(device)
        live_fake_labels = live_fake_labels.to(device)
        if current_iter % 100 == 0:
            first_iter = True  # 100번째 이터레이션마다 첫 번째 이터레이션으로 초기화
        # SAM 또는 표준 학습 스텝
        if use_sam:
            images = torch.cat([view1, view2], dim=0)
            logits, features = model(images, return_features=True)
            loss_ce = criterion(logits, torch.cat([live_fake_labels, live_fake_labels], dim=0))
            loss_con = lambda_contrastive * criterion_contrastive(features[:view1.size(0)], features[view1.size(0):], attack_ids)
            total_loss = loss_ce + loss_con
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.first_step(zero_grad=True)

            logits, features = model(images, return_features=True)
            loss_ce = criterion(logits, torch.cat([live_fake_labels, live_fake_labels], dim=0))
            loss_con = lambda_contrastive * criterion_contrastive(features[:view1.size(0)], features[view1.size(0):], attack_ids)
            total_loss = loss_ce +  loss_con
            total_loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            with autocast():
                optimizer.zero_grad()
                images = torch.cat([view1, view2], dim=0)
                logits, features = model(images, return_features=True)
                loss_ce = criterion(logits, torch.cat([live_fake_labels, live_fake_labels], dim=0))
                loss_con = lambda_contrastive * criterion_contrastive(features[:view1.size(0)], features[view1.size(0):], attack_ids)
                total_loss = loss_ce +  loss_con
                # total_loss.backward()
                scaler.scale(total_loss).backward()
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        if first_iter:
            logging.info(f"\nCurrent iteration: {current_iter}, batch size: {view1.size(0)}")
            # check if the batch size is correct
            ## check if the batch has same number of samples for each class
            logging.info(f"Batch size: {view1.size(0)}, Live samples: {live_fake_labels.sum().item()}, Fake samples: {(live_fake_labels == 0).sum().item()}")
            
            ## check if the distribution of attack types is correct
            unique_attack_ids, counts = torch.unique(attack_ids, return_counts=True)
            logging.info(f"Attack IDs: {unique_attack_ids}, Counts: {counts}")
            del counts, unique_attack_ids  # 메모리 해제
            logging.info(f"Iteration {current_iter}, logits live min: {logits[:, 1].min().item()}, logits live max: {logits[:, 1].max().item()}")
            logging.info(f"Iteration {current_iter}, logits fake min: {logits[:, 0].min().item()}, logits fake max: {logits[:, 0].max().item()}")
            logging.info(f"Iteration {current_iter}, logits [0]: {logits[0].detach().cpu().numpy()}, live_fake_labels: {live_fake_labels[0].item()}")
            logging.info(f"Iteration {current_iter}, logits [-1]: {logits[-1].detach().cpu().numpy()}, live_fake_labels: {live_fake_labels[-1].item()}")
            first_iter = False
            
        # 정확도 계산 및 로그 출력
        predicted_logit = logits[view1.size(0):]  # 첫 번째 뷰의 로짓만 사용
        
        _, predicted_labels = torch.max(predicted_logit.detach(), 1)
        acc = (predicted_labels == live_fake_labels).float().mean().item()
        pbar.set_postfix({"Total_Loss": f"{total_loss.item():.4f}", "Acc": f"{acc:.4f}", "CE_Loss": f"{loss_ce.item():.4f}", "Contrastive_Loss": f"{loss_con.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})

        # 메모리 정리
        
        del loss_ce, loss_con, total_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 최종 모델 저장 ---
    final_model_save_path = os.path.join(model_save_dir,finetune_strategy, 'clip_fas_final.pth')
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), final_model_save_path)
    logging.info(f"Final model saved to {final_model_save_path}")
    logging.info("Training finished.")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Train a contrastive model for FAS")
    parser.add_argument('--train_base_dir', type=str, default='/workspace/datasets/FAS2/Data-train_preprocessed', help='Base directory for training data')
    main(args=parser.parse_args())