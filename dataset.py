# dataset.py (Conditional Augmentation for Live Samples)

import os
import cv2
import random
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from augmentation import MoirePattern

class FASDataset(Dataset):
    def __init__(self, base_dir, data_list_path, is_train=True, image_size=256):
        self.base_dir = base_dir
        self.is_train = is_train
        self.image_size = image_size
        self.items = []
        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    label = -1
                    img_path = parts[0]
                else:
                    img_path, label = parts[0], int(parts[1])
                attack_code = '0_0_0'
                if self.is_train and len(parts) > 2:
                    attack_code = parts[2]
                self.items.append((img_path, label, attack_code))
        
        # --- SOLUTION: Define separate transform pipelines ---
        self.setup_transforms()
        
        # On-the-fly 공격 시뮬레이션용 증강 함수
        self.moire_attack = MoirePattern()

    def setup_transforms(self):
        """학습 및 검증, 그리고 레이블 유형에 따른 데이터 변환을 설정합니다."""
        if self.is_train:
            # --- 1. Fake 샘플 및 Pseudo-Fake 샘플을 위한 일반화 증강 ---
            self.fake_transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                ], p=0.1),
                A.ToGray(p=0.1),
                A.OneOf([
                    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
                    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussNoise(variance=(10.0, 50.0), p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ], p=0.1),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                ], p=0.1),
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
                                min_holes=1, min_height=8, min_width=8, p=0.3),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])
            
            # --- 2. 순수 Live 샘플을 위한 제한된 증강 ---
            self.live_transform = A.Compose([
                # Random Crop 대신, 이미지 크기를 약간 키운 후 랜덤하게 잘라내는 효과를 줌
                A.Resize(int(self.image_size * 1.1), int(self.image_size * 1.1)),
                A.RandomCrop(height=self.image_size, width=self.image_size, p=0.8),
                
                # 만약 crop이 적용 안되면 원래 사이즈로 복귀
                A.Resize(self.image_size, self.image_size),

                A.Rotate(limit=15, p=0.5), # -15도 ~ +15도 회전
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
                                min_holes=1, min_height=8, min_width=8, p=0.5), # Random Masking
                
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])

        # --- 3. 검증용 변환 (변경 없음) ---
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_rel_path, original_label, _ = self.items[idx]
        current_label = original_label
        
        img_full_path = os.path.join(self.base_dir, img_rel_path)
        
        try:
            image = cv2.imread(img_full_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {img_full_path}: {e}")
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        is_pseudo_fake = False
        # On-the-fly Attack Simulation (학습 & Live 샘플인 경우)
        if self.is_train and original_label == 1:
            aug_prob = random.random()
            if aug_prob < 0.2:
                image = self.moire_attack(image)
                current_label = 0
                is_pseudo_fake = True
            elif aug_prob < 0.4:
                mask_rel_path = img_rel_path.replace('images', 'masks').replace('.png', '.pkl')
                mask_full_path = os.path.join(self.base_dir, mask_rel_path)
                if os.path.exists(mask_full_path):
                    image = simulate_digital_swap(image, mask_full_path)
                    current_label = 0
                    is_pseudo_fake = True
        
        # --- SOLUTION: Apply transform based on the final label type ---
        if self.is_train:
            # 원본 Fake 또는 Pseudo-Fake 샘플인 경우
            if current_label == 0:
                transformed = self.fake_transform(image=image)
            # 순수 Live 샘플인 경우 (on-the-fly 증강이 적용되지 않음)
            else: # current_label == 1
                transformed = self.live_transform(image=image)
            image = transformed['image']
        else: # 검증 모드
            transformed = self.val_transform(image=image)
            image = transformed['image']
        
        return image, torch.tensor(current_label, dtype=torch.long)

    # ... (get_label, get_attack_code 메서드는 변경 없음) ...
    def get_label(self, idx):
        return self.items[idx][1]

    def get_attack_code(self, idx):
        if self.items[idx][1] == 1:
             return '0_0_0'
        return self.items[idx][2]
    
# dataset.py (Conditional Augmentation for Live Samples)


import os
import cv2
import random
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset

# 확장된 augmentation 모듈 임포트
from augmentation import (
    load_mask_and_get_parts,
    simulate_print, simulate_replay, simulate_cutouts, simulate_3d_mask,
    simulate_attribute_edit, simulate_faceswap, simulate_video_driven,
    simulate_pixel_level_adv, simulate_semantic_level_adv,
    simulate_id_consistent_gen, simulate_style_transfer_gen, simulate_prompt_driven_gen
)

class FASDatasetWithCLIP(Dataset):
    """
    기본 데이터셋. 이미지를 로드하고, 지시받은 공격을 적용하며, CLIP 전처리를 수행.
    """
    def __init__(self, base_dir, data_list_path):
        self.base_dir = base_dir
        self.items = []
        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                img_path, label = parts[0], int(parts[1])
                code = str(parts[2])
                self.items.append((img_path, label, code))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.attack_fn_map = self._get_attack_fn_map()

    def _get_attack_fn_map(self):
        # 공격 코드와 시뮬레이션 함수를 매핑
        return {
            '1_0_0': simulate_print, '1_0_1': simulate_replay, '1_0_2': simulate_cutouts,
            '1_1_0': lambda img, parts: simulate_3d_mask(img, parts, 'transparent'),
            '1_1_1': lambda img, parts: simulate_3d_mask(img, parts, 'plaster'),
            '1_1_2': lambda img, parts: simulate_3d_mask(img, parts, 'resin'),
            '2_0_0': simulate_attribute_edit, '2_0_1': simulate_faceswap, 
            '2_0_2': simulate_video_driven, '2_1_0': simulate_pixel_level_adv, 
            '2_1_1': simulate_semantic_level_adv, '2_2_0': simulate_id_consistent_gen, 
            '2_2_1': simulate_style_transfer_gen, '2_2_2': simulate_prompt_driven_gen,
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, instruction):
        """
        instruction: (원본 인덱스, 적용할 공격 코드) 튜플
        """
        original_idx, attack_code = instruction
        img_rel_path, original_label, original_code = self.items[original_idx]
        img_full_path = os.path.join(self.base_dir, img_rel_path)

        try:
            image_pil = Image.open(img_full_path).convert("RGB")
        except Exception as e:
            # print(f"Error reading image {img_full_path}: {e}")
            # 에러 발생 시, 유효한 다른 샘플을 재귀적으로 로드
            return self.__getitem__((random.randint(0, len(self.items) - 1), 'LIVE'))

        image_np = np.array(image_pil)
        
        # 지시받은 공격 적용
        if attack_code != '0_0_0' and original_label != 0:
            func = self.attack_fn_map.get(attack_code)
            if func:
                # 마스크가 필요한 함수인지 확인
                if attack_code in ['1_0_2', '1_1_0', '1_1_1', '1_1_2', '2_0_0', '2_0_2']:
                    mask_path = os.path.join(self.base_dir, img_rel_path.replace('images', 'masks').replace('.png', '.pkl'))
                    face_parts = load_mask_and_get_parts(mask_path)
                    if face_parts: image_np = func(image_np, face_parts)
                elif attack_code == '2_0_1': # faceswap은 mask_path를 직접 받음
                    mask_path = os.path.join(self.base_dir, img_rel_path.replace('images', 'masks').replace('.png', '.pkl'))
                    if os.path.exists(mask_path): image_np = func(image_np, mask_path)
                else: # 마스크 불필요
                    image_np = func(image_np)

        final_label = 0 if attack_code != '0_0_0' else 1
        
        image_pil_final = Image.fromarray(image_np)
        image_tensor = self.clip_preprocess(image_pil_final)
        
        return image_tensor, torch.tensor(final_label, dtype=torch.long)
    
# dataset.py (Corrected Logic)

import os
import random
import torch
import clip
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import inspect

# augmentation.py는 이전의 고도화된 버전을 그대로 사용
from augmentation import (
    load_mask_and_get_parts, simulate_print, simulate_replay, simulate_cutouts, 
    simulate_3d_mask, simulate_attribute_edit, simulate_faceswap, 
    simulate_video_driven, simulate_pixel_level_adv, simulate_semantic_level_adv, 
    simulate_id_consistent_gen, simulate_style_transfer_gen, simulate_prompt_driven_gen
)

class FASDatasetWithInstruction(Dataset):
    """
    기본 데이터셋. 샘플러로부터 (원본 인덱스, 적용할 공격 코드) 지시를 받아 처리.
    """
    def __init__(self, base_dir, data_list_path, is_train=True):
        self.base_dir = base_dir
        self.items = []
        self.samples_info = [] 
        self.is_train = is_train
        self.image_size = 224  # CLIP 모델에 맞춘 이미지 크기
        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                # 경로, 코드, 레이블 형식으로 된 데이터 파일을 가정
                if len(parts) < 3: continue 
                
                img_path, label, attack_code = parts[0], int(parts[1]), parts[2]
                self.items.append((img_path, label))
                self.samples_info.append({'path': img_path, 'label': label, 'code': attack_code})

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attack_fn_map = self._get_attack_fn_map()
        if self.is_train:
            # --- 2. 순수 Live 샘플을 위한 제한된 증강 ---
            self.patch_transform = A.Compose([
                    # --- 기본 변형 (거의 항상 적용) ---
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.3, shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
                    # --- 색상 및 밝기 변화 (다양한 조명/센서 모방) ---
                    A.OneOf([
                        A.RandomBrightnessContrast(p=1.0, brightness_limit=0.15, contrast_limit=0.15),
                        A.HueSaturationValue(p=1.0, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15),
                        A.ColorJitter(p=1.0, brightness=0.15, contrast=0.15, saturation=0.2, hue=0.1)
                    ], p=0.7),
                    # --- 카메라 아티팩트 (물리적 특성 모방) ---
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 10), p=1.0),
                        A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
                    ], p=0.5),
                    # --- 센서 노이즈 ---
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    ], p=0.5),
                    # --- 렌즈 및 환경 효과 (가장 복잡한 변화) ---
                    A.OneOf([
                        A.OpticalDistortion(p=1.0, distort_limit=0.1, shift_limit=0.1),
                        A.GridDistortion(p=1.0),
                        A.RandomSunFlare(p=1.0, flare_roi=(0, 0, 1, 0.5), src_radius=100),
                        A.RandomShadow(p=1.0)
                    ], p=0.2),
                    # --- 부분 가려짐 ---
                    A.CoarseDropout(p=0.1, max_holes=3, max_height=25, max_width=25, min_height=10, min_width=10, fill_value=0),      
                    A.RandomResizeCrop(height=self.image_size, width=self.image_size, scale=(0.4, 0.8), p=0.5),
                    # --- 최종 정규화 및 텐서 변환 ---
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2(),
                    ])
            self.full_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.3, shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT),
                    # --- 색상 및 밝기 변화 (다양한 조명/센서 모방) ---
                    A.OneOf([
                        A.RandomBrightnessContrast(p=1.0, brightness_limit=0.15, contrast_limit=0.15),
                        A.HueSaturationValue(p=1.0, hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15),
                        A.ColorJitter(p=1.0, brightness=0.15, contrast=0.15, saturation=0.2, hue=0.1)
                    ], p=0.7),
                    # --- 카메라 아티팩트 (물리적 특성 모방) ---
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 10), p=1.0),
                        A.Downscale(scale_min=0.7, scale_max=0.9, p=1.0),
                    ], p=0.5),
                    # --- 센서 노이즈 ---
                    A.OneOf([
                        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    ], p=0.5),
                    # --- 렌즈 및 환경 효과 (가장 복잡한 변화) ---
                    A.OneOf([
                        A.OpticalDistortion(p=1.0, distort_limit=0.1, shift_limit=0.1),
                        A.GridDistortion(p=1.0),
                        A.RandomSunFlare(p=1.0, flare_roi=(0, 0, 1, 0.5), src_radius=100),
                        A.RandomShadow(p=1.0)
                    ], p=0.2),
                    # --- 부분 가려짐 ---
                    A.CoarseDropout(p=0.1, max_holes=3, max_height=25, max_width=25, min_height=10, min_width=10, fill_value=0),      
                    # --- 최종 정규화 및 텐서 변환 ---
                    
                    A.RandomResizeCrop(height=self.image_size, width=self.image_size, scale=(0.9, 1.0), p=0.5),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2(),
                    ])
        self.moire_attack = MoirePattern()
        # --- 3. 검증용 변환 (변경 없음) ---
        self.val_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def _get_attack_fn_map(self):
        return {
            '1_0_0': simulate_print,
            '1_0_1': simulate_replay,
            '1_0_2': simulate_cutouts,
            '2_0_0': lambda img, parts: simulate_attribute_edit(img, parts),
            '2_0_1': lambda img, parts: simulate_faceswap(img, parts),
            '2_0_2': lambda img, parts: simulate_video_driven(img, parts),
            '2_1_0': lambda img, parts=None: simulate_pixel_level_adv(img, parts),
            '2_1_1': lambda img, parts=None: simulate_semantic_level_adv(img, parts),
        }
        
    def __len__(self):
        return len(self.items)
    def base_attack_code(self, img_np):
        """
        이미지에 기본 공격 코드를 적용합니다.
        patch 단위로 변형한 후, 원본 이미지와 합성합니다.
        이미지를 9x9 패치로 나누고, 각 패치에 대해 augmentation을 적용합니다.
        """
        patch_size = 32
        h, w, _ = img_np.shape
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img_np[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue
                # Moire 패턴 공격 적용
                patch = self.moire_attack(patch)
                img_np[i:i+patch_size, j:j+patch_size] = patch
        return img_np
        
        
    def __getitem__(self, instruction):
        original_idx, attack_code_to_apply = instruction
        img_rel_path, _ = self.items[original_idx]
        img_full_path = os.path.join(self.base_dir, img_rel_path)
        try:
            image_pil = Image.open(img_full_path).convert("RGB")
        except Exception:
            # 에러 발생 시, 유효한 다른 Live 샘플을 재귀적으로 로드
            valid_live_indices = [i for i, info in enumerate(self.samples_info) if info['label'] == 1]
            if not valid_live_indices: return torch.zeros(3, 224, 224), torch.tensor(0, dtype=torch.long) # Fallback
            return self.__getitem__((random.choice(valid_live_indices), 'LIVE'))

        image_np = np.array(image_pil)
        
        # 'LIVE' 또는 'REAL_FAKE'가 아니면 pseudo-attack 적용
        if attack_code_to_apply not in ['LIVE', 'REAL_FAKE']:
            func = self.attack_fn_map.get(attack_code_to_apply)
            if func:
                sig = inspect.signature(func)
                mask_path = os.path.join(self.base_dir, img_rel_path.replace('images', 'masks').replace('.png', '.pkl'))
                
                if len(sig.parameters) > 1:
                    if 'face_parts' in sig.parameters:
                        face_parts = load_mask_and_get_parts(mask_path)
                        if not face_parts:
                            # 마스크가 없으면, 아무 공격이라도 적용
                            image_np = self.base_attack_code(image_np)
                        if face_parts: image_np = func(image_np, face_parts)
                    elif 'mask_path' in sig.parameters:
                        if os.path.exists(mask_path): image_np = func(image_np, mask_path)
                        else:
                            image_np = self.base_attack_code(image_np)
                else:
                    image_np = func(image_np)
        
        # 'LIVE' 지시일 때만 레이블 1, 나머지는 모두 0
        final_label = 1 if attack_code_to_apply == 'LIVE' else 0
        
        
        if self.is_train:
            if final_label == 0:
                transformed = self.fake_transform(image=image_np)
            else:  # final_label == 1
                transformed = self.live_transform(image=image_np)
            image_tensor = transformed['image']
        else:  # 검증 모드
            transformed = self.val_transform(image=image_np)
            image_tensor = transformed['image']
        # CLIP 전처리 적용
        
        return image_tensor, torch.tensor(final_label, dtype=torch.long)
    
    
class FASValidationDataset(Dataset):
    """
    검증용 데이터셋. 공격 유형(attack_code)과 레이블(label)을 함께 반환합니다.
    """
    def __init__(self, base_dir, data_list_path):
        self.base_dir = base_dir
        self.items = []
        self.samples_info = []
        self.image_size = 224  # CLIP 모델에 맞춘 이미지 크기

        # 데이터 리스트 파일 읽기
        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3:
                    self.items.append((parts[0], -1, '0_0_0'))  # 레이블이 없는 경우
                    continue  # 형식이 맞지 않는 경우 건너뜀
                
                img_path, label, attack_code = parts[0], int(parts[1]), parts[2]
                
                self.items.append((img_path, label, attack_code))
                self.samples_info.append({
                    'path': img_path,
                    'label': label,
                    'code': attack_code
                })

        # 검증용 변환 (증강 없음)
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
        """
        Data label description

        ********** Live **********
        *** 0_0_0: Live Face

        ********** Fake **********
        *   1_   : Physical Attack
        **  1_0_ : 2D Attack
        *** 1_0_0: Print
        *** 1_0_1: Replay
        *** 1_0_2: Cutouts
        **  1_1_ : 3D Attack
        *** 1_1_0: Transparent
        *** 1_1_1: Plaster
        *** 1_1_2: Resin

        *   2_   : Digital Attack
        **  2_0_ : Digital Manipulation
        *** 2_0_0: Attibute-Edit
        *** 2_0_1: Face-Swap
        *** 2_0_2: Video-Driven
        **  2_1_ : Digital Adversarial
        *** 2_1_0: Pixcel-Level
        *** 2_1_1: Semantic-Level
        **  2_2_ : Digital Generation
        *** 2_2_0: ID_Consisnt
        *** 2_2_1: Style
        *** 2_2_2: Prompt
        """
        self.attack_type_map = {
            '0_0_0': 0,
            '1_0_0': 1,
            '1_0_1': 2,
            '1_0_2': 3,
            '1_1_0': 4,
            '1_1_1': 5,
            '1_1_2': 6,
            '2_0_0': 7,
            '2_0_1': 8,
            '2_0_2': 9,
            '2_1_0': 10,
            '2_1_1': 11,
            '2_2_0': 12,
            '2_2_1': 13,
            '2_2_2': 14,
        }
        self.attack_int_to_str = {
            0: 'LIVE',
            1: 'Physical Attack: 2D/Print',
            2: 'Physical Attack: 2D/Replay',
            3: 'Physical Attack: 2D/Cutouts',
            4: 'Physical Attack: 3D/Transparent',
            5: 'Physical Attack: 3D/Plaster',
            6: 'Physical Attack: 3D/Resin',
            7: 'Digital Attack: Attibute-Edit',
            8: 'Digital Attack: Face-Swap',
            9: 'Digital Attack: Video-Driven',
            10: 'Digital Adversarial: Pixel-Level',
            11: 'Digital Adversarial: Semantic-Level',
            12: 'Digital Generation: ID-Consisnt',
            13: 'Digital Generation: Style',
            14: 'Digital Generation: Prompt',
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, label, attack_code = self.items[idx]
        img_full_path = os.path.join(self.base_dir, img_path)
        image_pil = Image.open(img_full_path).convert("RGB")
        image_np = np.array(image_pil)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # 공격 유형을 정수로 매핑
        attack_type = self.attack_type_map[attack_code]
        return image_tensor, torch.tensor(label, dtype=torch.long), torch.tensor(attack_type, dtype=torch.long)
    
class FASSimCLRDataset(Dataset):
    def __init__(self, base_dir, data_list_path, cache=False):
        self.base_dir = base_dir
        self.samples_info = [] 
        self.image_size = 224
        
        self.cache = {}
        self.use_cache = cache
        self.attack_code_to_id, self.id_to_attack_code = self._get_attack_id_maps()

        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3: continue 
                img_path, label, attack_code = parts[0], int(parts[1]), parts[2]
                self.samples_info.append({
                    'path': img_path, 'label': label, 'code': attack_code,
                    'attack_id': self.attack_code_to_id.get(attack_code)
                })

        self.attack_fn_map = self._get_attack_fn_map()
        self._setup_transforms()
        self.moire_attack = MoirePattern()
    def base_attack_code(self, img_np):
        """
        이미지에 기본 공격 코드를 적용합니다.
        patch 단위로 변형한 후, 원본 이미지와 합성합니다.
        이미지를 9x9 패치로 나누고, 각 패치에 대해 augmentation을 적용합니다.
        """
        patch_size = 32
        h, w, _ = img_np.shape
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img_np[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue
                attack_id = random.choice([0,1,2,3,4])
                if attack_id == 0:
                    # Moire 패턴 공격 적용
                    patch = self.moire_attack(patch)
                elif attack_id == 1:
                    # Resize 후 다시 Upscale
                    patch_size_ = patch.shape[0] // 2
                    patch = cv2.resize(patch, (patch_size_, patch_size_))
                    patch = cv2.resize(patch, (patch_size, patch_size))
                elif attack_id == 2:
                    # Noise 추가
                    noise = np.random.normal(0, 0.1, patch.shape).astype(patch.dtype)
                    patch = cv2.addWeighted(patch, 0.5, noise, 0.5, 0)
                elif attack_id == 3:
                    # Random Rotation
                    angle = random.randint(-90, 90)
                    M = cv2.getRotationMatrix2D((patch_size/2, patch_size/2), angle, 1)
                    patch = cv2.warpAffine(patch, M, (patch_size, patch_size), flags=cv2.INTER_LINEAR)
                elif attack_id == 4:
                    pass # No attack, just keep the original patch
                img_np[i:i+patch_size, j:j+patch_size] = patch
        return img_np
        
    def _get_attack_id_maps(self):
        codes = ['0_0_0', '1_0_0', '1_0_1', '1_0_2', '1_1_0', '1_1_1', '1_1_2',
                 '2_0_0', '2_0_1', '2_0_2', '2_1_0', '2_1_1'] # 사용 가능한 공격 코드
        code_to_id = {code: i for i, code in enumerate(codes)}
        id_to_code = {i: code for i, code in enumerate(codes)}
        return code_to_id, id_to_code
    def _get_attack_fn_map(self):
        return {
            '1_0_0': simulate_print,
            '1_0_1': simulate_replay,
            '1_0_2': simulate_cutouts,
            # '1_1_0': lambda img, parts: simulate_3d_mask(img, parts, mode='transparent'),
            # '1_1_1': lambda img, parts: simulate_3d_mask(img, parts, mode='plaster'),
            # '1_1_2': lambda img, parts: simulate_3d_mask(img, parts, mode='resin'),
            '2_0_0': lambda img, parts: simulate_attribute_edit(img, parts),
            '2_0_1': lambda img, parts: simulate_faceswap(img, parts),
            '2_0_2': lambda img, parts: simulate_video_driven(img, parts),
            '2_1_0': lambda img, parts=None: simulate_pixel_level_adv(img, parts),
            '2_1_1': lambda img, parts=None: simulate_semantic_level_adv(img, parts),
            # '2_2_0': simulate_id_consistent_gen,
            # '2_2_1': simulate_style_transfer_gen,
            # '2_2_2': simulate_prompt_driven_gen,
        }
    def _setup_transforms(self):
        self.view2_transform = A.Compose([
                A.Resize(height=int(self.image_size * 1.1), width=int(self.image_size * 1.1), p=1.0),
                A.OneOf([
                    A.Resize(height=int(self.image_size * 1.5), width=int(self.image_size * 1.5), p=1.0),
                    A.Resize(height=self.image_size * 2, width=self.image_size * 2, p=1.0),
                ], p=0.5),
                A.CoarseDropout(
                max_holes=3,
                max_height=25,
                max_width=25,
                min_height=10,
                min_width=10,
                fill_value=0,
                p=0.5
                ),
                A.OneOf([
                    A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.02, 0.1), p=1.0),
                ], p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
                ], p=0.2),
                A.Rotate(limit=15, p=0.5), # -15도 ~ +15도 회전
                A.RandomCrop(height=self.image_size, width=self.image_size, p=1.0),
                A.Resize(height=self.image_size, width=self.image_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0),
                A.pytorch.transforms.ToTensorV2()
            ])
        self.view1_transform = A.Compose([
                A.Resize(height=int(self.image_size * 1.1), width=int(self.image_size * 1.1), p=0.8),
                A.Resize(height=self.image_size * 1.5, width=self.image_size * 1.5, p=0.8),
                A.CoarseDropout(
                max_holes=3,
                max_height=25,
                max_width=25,
                min_height=10,
                min_width=10,
                fill_value=0,
                p=0.5
                ),
                A.OneOf([
                    A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.02, 0.1), p=1.0),
                ], p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
                ], p=0.2),
                A.Rotate(limit=15, p=0.5), # -15도 ~ +15도 회전
                A.RandomCrop(height=self.image_size, width=self.image_size, p=0.8),
                A.Resize(height=self.image_size, width=self.image_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0),
                A.pytorch.transforms.ToTensorV2()
            ])
        self.fake_transform = A.Compose([
                # structure 관련 augmentation 
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # color 관련 augmentation
                A.OneOf([
                    A.Sharpen(),
                    A.Emboss(),
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ], p=0.2),
                A.ToGray(p=0.2),
                # compression 관련 augmentation
                A.OneOf([
                    A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="jpeg", p=0.5),
                    A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="webp", p=0.5),
                ], p=0.2),
                # perturbation 관련 augmentation
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ], p=0.2),
            ])

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, instruction, depth=0):
        original_idx, action, attack_id_to_apply = instruction
        info = self.samples_info[original_idx]
        img_full_path = os.path.join(self.base_dir, info['path'])
        if depth > 10:
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        try:
            if self.use_cache and img_full_path in self.cache:
                image_np = self.cache[img_full_path]
            else:
                image_np = np.array(Image.open(img_full_path).convert("RGB"))
                if self.use_cache:
                    self.cache[img_full_path] = image_np
        except Exception:
            # 에러 발생 시, 다른 Live 이미지로 대체하여 Pseudo-Fake 생성
            live_indices = [i for i, s in enumerate(self.samples_info) if s['label'] == 1]
            return self.__getitem__((random.choice(live_indices), 'PSEUDO_FAKE', attack_id_to_apply), depth + 1)

        if action == 'PSEUDO_FAKE':
            try:
                attack_code = self.id_to_attack_code.get(attack_id_to_apply)
                if attack_code:
                    func = self.attack_fn_map.get(attack_code)
                    if func:
                        sig = inspect.signature(func)
                        if len(sig.parameters) > 1:
                            if 'face_parts' in sig.parameters:
                                mask_path = os.path.join(self.base_dir, info['path'].replace('images', 'masks').replace('.png', '.pkl'))
                                face_parts = load_mask_and_get_parts(mask_path)
                                if face_parts: image_np = func(image_np, face_parts)
                                else:
                                    image_np = self.base_attack_code(image_np)  # 기본 공격 적용
                            elif 'mask_path' in sig.parameters:
                                mask_path = os.path.join(self.base_dir, info['path'].replace('images', 'masks').replace('.png', '.pkl'))
                                if os.path.exists(mask_path): image_np = func(image_np, mask_path)
                                else:
                                    image_np = self.base_attack_code(image_np)
                        else:
                            image_np = func(image_np)
            except Exception as e:
                print(f"Error applying attack {attack_code} on image {img_full_path}: {e}",flush=True)
                image_np = self.base_attack_code(image_np)
        elif action == 'LIVE':
            # Live 샘플은 변형 없이 그대로 사용
            pass
        elif action == 'REAL_FAKE':
            # 실제 Fake 샘플은 Moire 패턴 공격 적
            image_np = self.fake_transform(image=image_np)['image']
        else:
            raise ValueError(f"Unknown action: {action}")
        final_live_fake_label = 1 if action == 'LIVE' else 0
        final_attack_id = attack_id_to_apply
        if image_np.shape[0] < self.image_size or image_np.shape[1] < self.image_size:
            # 이미지 크기가 너무 작으면, 회색 배경으로 채움
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            # 이미지 중간에 원본 이미지를 배치
            h, w, _ = image_np.shape
            start_h = (h - image_np.shape[0]) // 2
            start_w = (w - image_np.shape[1]) // 2
            image_np[start_h:start_h+image_np.shape[0], start_w:start_w+image_np.shape[1]] = image_np
            
        view1_tensor = self.view1_transform(image=image_np)['image']
        view2_tensor = self.view2_transform(image=image_np)['image']
        
        return (view1_tensor, view2_tensor, 
                torch.tensor(final_live_fake_label, dtype=torch.long), 
                torch.tensor(final_attack_id, dtype=torch.long))


        
class FASADVDataset(Dataset):
    def __init__(self, base_dir, data_list_path, cache=False):
        self.base_dir = base_dir
        self.samples_info = [] 
        self.image_size = 224
        
        self.cache = {}
        self.use_cache = cache
        self.attack_code_to_id, self.id_to_attack_code = self._get_attack_id_maps()

        with open(data_list_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 3: continue 
                img_path, label, attack_code = parts[0], int(parts[1]), parts[2]
                self.samples_info.append({
                    'path': img_path, 'label': label, 'code': attack_code,
                    'attack_id': self.attack_code_to_id.get(attack_code)
                })

        self.attack_fn_map = self._get_attack_fn_map()
        self._setup_transforms()
        self.moire_attack = MoirePattern()
    def base_attack_code(self, img_np):
        """
        이미지에 기본 공격 코드를 적용합니다.
        patch 단위로 변형한 후, 원본 이미지와 합성합니다.
        이미지를 9x9 패치로 나누고, 각 패치에 대해 augmentation을 적용합니다.
        """
        patch_size = 32
        h, w, _ = img_np.shape
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img_np[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue
                attack_id = random.choice([0,1,2,3,4])
                if attack_id == 0:
                    # Moire 패턴 공격 적용
                    patch = self.moire_attack(patch)
                elif attack_id == 1:
                    # Resize 후 다시 Upscale
                    patch_size_ = patch.shape[0] // 2
                    patch = cv2.resize(patch, (patch_size_, patch_size_))
                    patch = cv2.resize(patch, (patch_size, patch_size))
                elif attack_id == 2:
                    # Noise 추가
                    noise = np.random.normal(0, 0.1, patch.shape).astype(patch.dtype)
                    patch = cv2.addWeighted(patch, 0.5, noise, 0.5, 0)
                    del noise
                elif attack_id == 3:
                    # Random Rotation
                    angle = random.randint(-90, 90)
                    M = cv2.getRotationMatrix2D((patch_size/2, patch_size/2), angle, 1)
                    patch = cv2.warpAffine(patch, M, (patch_size, patch_size), flags=cv2.INTER_LINEAR)
                    del M, angle
                elif attack_id == 4:
                    pass # No attack, just keep the original patch
                img_np[i:i+patch_size, j:j+patch_size] = patch
        
        return img_np
        
    def _get_attack_id_maps(self):
        codes = ['0_0_0', '1_0_0', '1_0_1', '1_0_2', '1_1_0', '1_1_1', '1_1_2',
                 '2_0_0', '2_0_1', '2_0_2', '2_1_0', '2_1_1'] # 사용 가능한 공격 코드
        code_to_id = {code: i for i, code in enumerate(codes)}
        id_to_code = {i: code for i, code in enumerate(codes)}
        return code_to_id, id_to_code
    def _get_attack_fn_map(self):
        return {
            '1_0_0': simulate_print,
            '1_0_1': simulate_replay,
            '1_0_2': simulate_cutouts,
            '2_0_0': lambda img, parts: simulate_attribute_edit(img, parts),
            '2_0_1': lambda img, parts: simulate_faceswap(img, parts),
            '2_0_2': lambda img, parts: simulate_video_driven(img, parts),
            '2_1_0': lambda img, parts=None: simulate_pixel_level_adv(img, parts),
            '2_1_1': lambda img, parts=None: simulate_semantic_level_adv(img, parts),
        }
    def get_attack_code(self, idx):
        if self.samples_info[idx]['code'] not in self.attack_code_to_id:
             return '0_0_0'
        return self.samples_info[idx]['code']
    def get_label(self, idx):
        return self.samples_info[idx]['label']
    def _setup_transforms(self):
        self.view1_transform = A.Compose([
                A.Resize(height=int(self.image_size * 1.1), width=int(self.image_size * 1.1), p=1.0),
                A.Resize(height=self.image_size * 1.5, width=self.image_size * 1.5, p=0.5),
                A.CoarseDropout(
                max_holes=3,
                max_height=25,
                max_width=25,
                min_height=10,
                min_width=10,
                fill_value=0,
                p=0.5
                ),
                A.OneOf([
                    A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.02, 0.1), p=1.0),
                ], p=0.2),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
                ], p=0.2),
                A.Rotate(limit=15, p=0.5), # -15도 ~ +15도 회전
                A.RandomCrop(height=self.image_size, width=self.image_size, p=0.8),
                A.Resize(height=self.image_size, width=self.image_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, p=1.0),
                A.pytorch.transforms.ToTensorV2()
            ])
        self.fake_transform = A.Compose([
                # structure 관련 augmentation 
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # color 관련 augmentation
                A.OneOf([
                    A.Sharpen(),
                    A.Emboss(),
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ], p=0.2),
                A.ToGray(p=0.2),
                # compression 관련 augmentation
                A.OneOf([
                    A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="jpeg", p=0.5),
                    A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="webp", p=0.5),
                ], p=0.2),
                # perturbation 관련 augmentation
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.MotionBlur(blur_limit=(3, 7), p=0.5),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                ], p=0.2),
            ])

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, instruction, depth=0):
        original_idx, action, attack_id_to_apply = instruction
        info = self.samples_info[original_idx]
        img_full_path = os.path.join(self.base_dir, info['path'])
        if depth > 10:
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        try:
            if self.use_cache and img_full_path in self.cache:
                image_np = self.cache[img_full_path]
            else:
                image_np = np.array(Image.open(img_full_path).convert("RGB"))
                if self.use_cache:
                    self.cache[img_full_path] = image_np
        except Exception:
            # 에러 발생 시, 다른 Live 이미지로 대체하여 Pseudo-Fake 생성
            live_indices = [i for i, s in enumerate(self.samples_info) if s['label'] == 1]
            return self.__getitem__((random.choice(live_indices), 'PSEUDO_FAKE', attack_id_to_apply), depth + 1)

        if action == 'PSEUDO_FAKE':
            try:
                attack_code = self.id_to_attack_code.get(attack_id_to_apply)
                if attack_code:
                    func = self.attack_fn_map.get(attack_code)
                    if func:
                        sig = inspect.signature(func)
                        if len(sig.parameters) > 1:
                            if 'face_parts' in sig.parameters:
                                mask_path = os.path.join(self.base_dir, info['path'].replace('images', 'masks').replace('.png', '.pkl'))
                                face_parts = load_mask_and_get_parts(mask_path)
                                if face_parts: 
                                    image_np = func(image_np, face_parts)
                                    del face_parts
                                else:
                                    image_np = self.base_attack_code(image_np)  # 기본 공격 적용
                            elif 'mask_path' in sig.parameters:
                                mask_path = os.path.join(self.base_dir, info['path'].replace('images', 'masks').replace('.png', '.pkl'))
                                if os.path.exists(mask_path): image_np = func(image_np, mask_path)
                                else:
                                    image_np = self.base_attack_code(image_np)
                        else:
                            image_np = func(image_np)
            except Exception as e:
                print(f"Error applying attack {attack_code} on image {img_full_path}: {e}",flush=True)
                image_np = self.base_attack_code(image_np)
            
        elif action == 'LIVE':
            # Live 샘플은 변형 없이 그대로 사용
            pass
        elif action == 'REAL_FAKE':
            # 실제 Fake 샘플은 Moire 패턴 공격 적
            image_np = self.fake_transform(image=image_np)['image']
        else:
            raise ValueError(f"Unknown action: {action}")
        final_live_fake_label = 1 if action == 'LIVE' else 0
        final_attack_id = attack_id_to_apply
        # view1_tensor = self.view1_transform(image=image_np)['image']
            
        view1_tensor = self.view1_transform(image=image_np)['image']
        # view2_tensor = self.view2_transform(image=image_np)['image']
        
        
        return (view1_tensor, 
                torch.tensor(final_live_fake_label, dtype=torch.long), 
                torch.tensor(final_attack_id, dtype=torch.long))

