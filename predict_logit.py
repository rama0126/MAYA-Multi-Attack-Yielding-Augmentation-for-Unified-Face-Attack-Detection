# predict.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse

# 로컬 모듈 임포트
from dataset import FASValidationDataset
from model import ClipFas
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 사용할 GPU 설정
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_and_save(model, dataloader, device, output_file, is_val=False):
    """
    모델을 사용하여 데이터셋을 예측하고 결과를 파일에 저장합니다.
    """
    model.eval()  # 모델을 평가 모드로 설정
    all_probs = []
    all_labels = []
    all_atks = []

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

    # --- 결과 파일 저장 ---
    # dataloader의 dataset 객체에서 이미지 경로 리스트를 가져옴
    image_paths = [item[0] for item in dataloader.dataset.items]
        # 최종 결과를 numpy 배열로 변환
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_atks = np.array(all_atks)
    # 원본 데이터 경로 형식에 맞게 수정 ('images/' 접두사 제거)
    original_paths = [path.replace('images/', '') for path in image_paths]
    if is_val:
        from evaluate import find_optimal_threshold, calculate_metrics
        optimal_threshold, optimal_acer = find_optimal_threshold(all_labels, all_probs)
        logging.info(f"Optimal Threshold: {optimal_threshold:.4f}, Optimal ACER: {optimal_acer:.4f}")
        logging.info(f"Validation dataset size: {len(original_paths)}")
        metrics = calculate_metrics(all_labels, all_probs, threshold=optimal_threshold)
        logging.info(f"Validation metrics: {metrics}")
        unique_attack_types = np.unique(all_atks)
        live_mask = all_labels == 1
        attack_int_to_str = dataloader.dataset.attack_int_to_str if hasattr(dataloader.dataset, 'attack_int_to_str') else attack_int_to_str
        for attack_type in unique_attack_types:
            attack_mask = all_atks == attack_type
            if np.sum(attack_mask) == 0:
                continue  # 해당 공격 유형에 대한 샘플이 없는 경우 건너뜀
            # mask = attack_mask U live_mask
            mask = attack_mask | live_mask
            attack_labels = all_labels[mask]
            attack_probs = all_probs[mask]
            attack_metrics = calculate_metrics(attack_labels, attack_probs, threshold=optimal_threshold)
            attack_type_name = attack_int_to_str[attack_type] if attack_int_to_str else f'ATTACK_{attack_type}'
            logging.info(f"Attack Type: {attack_type_name}, Metrics: {attack_metrics}")
    with open(output_file, 'w') as f:
        for path, score in zip(original_paths, all_probs):
            # score를 소수점 5자리까지 포맷팅
            f.write(f"{path} {score:.5f}\n")
            
    logging.info(f"Predictions saved to {output_file}")


def main(args):
    # --- 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # --- 모델 로드 ---
    logging.info(f"Loading model from {args.model_path}")
    # 모델 구조는 학습 때와 동일해야 함
    
    model = ClipFas(
        num_classes=2, 
        model_name='ViT-L-14', 
        pretrained='laion2b_s32b_b82k',
        finetune_strategy=args.finetune_strategy
    )
    
    # DataParallel로 학습했다면, state_dict의 키 이름이 다를 수 있음
    state_dict = torch.load(args.model_path, map_location=device)
    
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
    
    # 멀티 GPU 추론 (선택 사항, 배치 크기가 크면 유용)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for prediction.")
        model = torch.nn.DataParallel(model)

    # --- 검증 데이터로더 설정 ---
    logging.info("Setting up validation data loader...")
    # is_train=False로 설정하여 on-the-fly 증강을 비활성화
    val_dataset = FASValidationDataset(base_dir=os.path.join(args.base_dir, 'Data-val_preprocessed'),
                                       data_list_path=os.path.join(args.base_dir, 'Data-val_preprocessed', 'val_list.txt'),)
    if len(val_dataset) == 0:
        logging.error("Validation dataset is empty. Please check paths.")
        return
        
    # shuffle=False는 예측 순서와 파일 리스트 순서를 일치시키기 위해 필수
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    val_output_file = args.output_file.format(model_path=os.path.basename(args.model_path), type='val')
    # --- 예측 및 저장 실행 ---
    predict_and_save(model, val_loader, device, val_output_file, is_val=True)
    
    logging.info("Setting up test data loader...")
    # is_train=False로 설정하여 on-the-fly 증강을 비활성화
    test_dataset = FASValidationDataset(base_dir=os.path.join(args.base_dir, 'Data-test_preprocessed'),
                                       data_list_path=os.path.join(args.base_dir, 'Data-test_preprocessed','test_list.txt'),)
    if len(val_dataset) == 0:
        logging.error("Validation dataset is empty. Please check paths.")
        return
        
    # shuffle=False는 예측 순서와 파일 리스트 순서를 일치시키기 위해 필수
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_output_file = args.output_file.format(model_path=os.path.basename(args.model_path), type='test')
    # --- 예측 및 저장 실행 ---
    predict_and_save(model, test_loader, device, test_output_file)
    
    # merge validation and test results
    logging.info("Merging validation and test results...")
    with open(val_output_file, 'r') as val_file, open(test_output_file, 'r') as test_file, open(args.output_file.format(model_path=os.path.basename(args.model_path), type='merged'), 'w') as submission_file:
        val_lines = val_file.readlines()
        test_lines = test_file.readlines()
        
        # validation 결과를 먼저 작성
        submission_file.writelines(val_lines)
        
        # test 결과를 이어서 작성
        submission_file.writelines(test_lines)
    logging.info(f"Submission file created: submission_{os.path.basename(args.model_path)}_merged.txt")
    logging.info("Prediction finished.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict on the validation set and generate submission file.")
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (.pth).')
    parser.add_argument('--base_dir', type=str, default='/workspace/datasets/FAS3',)

    
    parser.add_argument('--output_file', type=str, default='submission_{model_path}_{type}.txt',
                        help='Path to save the output prediction file.')
                        
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size used for model input (must match training).')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for prediction.')
                        
    parser.add_argument('--finetune_strategy', type=str, default='Swin',
                        choices=['adapter', 'partial', 'linear_probe', 'full', 'Swin'],
                        help='Fine-tuning strategy used during training.')
    args = parser.parse_args()
    
    main(args)