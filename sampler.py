# sampler.py (Epoch Length Calculation Fixed)

import torch
import torch.utils.data
import numpy as np
import logging
from torch.utils.data import Sampler, Dataset
from typing import Iterator, List
import math
import random

from collections import defaultdict
class AttackTypeBatchSampler(Sampler[List[int]]):
    """
    Live와 모든 개별 공격 유형에 대해 균등하게 샘플링하는 샘플러.
    에포크 길이는 가장 많은 샘플을 가진 클래스를 기준으로 합니다.
    """
    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.attack_codes = np.array([self.dataset.get_attack_code(i) for i in range(len(self.dataset))])
        
        self.class_indices = {}
        self.unique_codes = np.unique(self.attack_codes)
        for code in self.unique_codes:
            self.class_indices[code] = np.where(self.attack_codes == code)[0]

        self.num_classes = len(self.unique_codes)
        
        if self.batch_size % self.num_classes != 0:
            raise ValueError(f"Batch size ({self.batch_size}) must be divisible by the number of attack types ({self.num_classes}).")
            
        self.samples_per_class_in_batch = self.batch_size // self.num_classes
        
        # --- SOLUTION: Calculate epoch length based on the LARGEST class ---
        # 이 방법을 통해 모든 데이터를 최소 한 번 이상 활용하게 됩니다.
        max_class_size = max(len(v) for v in self.class_indices.values())
        
        # 전체 배치의 수는 `(max_class_size / samples_per_class_in_batch)`로 결정
        num_batches_per_epoch = (max_class_size + self.samples_per_class_in_batch - 1) // self.samples_per_class_in_batch
        self.length = num_batches_per_epoch * self.batch_size

        logging.info(f"AttackTypeBatchSampler initialized.")
        logging.info(f"Found {self.num_classes} unique attack types: {self.unique_codes}")
        logging.info(f"Samples per attack type in batch: {self.samples_per_class_in_batch}")
        logging.info(f"Total samples per epoch: {self.length}")
        logging.info(f"Number of batches per epoch: {num_batches_per_epoch}")

    def __iter__(self) -> Iterator[List[int]]:
        # 각 클래스(공격 유형)의 인덱스를 셔플
        shuffled_class_indices = {k: np.random.permutation(v) for k, v in self.class_indices.items()}
        pointers = {k: 0 for k in self.class_indices}
        
        num_batches = self.length // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            for code in sorted(self.unique_codes):
                indices_for_code = self.class_indices[code]
                
                # --- SOLUTION: Use np.random.choice for robust over-sampling ---
                # 필요한 만큼 샘플을 뽑되, 샘플 수가 부족하면 복원 추출 (replace=True)
                num_to_sample = self.samples_per_class_in_batch
                
                # np.random.choice는 항상 복원추출을 하므로, 
                # 한 에포크 내에서 같은 샘플이 여러번 뽑힐 수 있어 다양성이 감소할 수 있음
                # 따라서, 인덱스를 반복 사용하는 방식으로 수정
                
                start = pointers[code]
                end = start + num_to_sample
                
                # 인덱스 리스트를 순환하도록 함
                batch_part = []
                class_idx_list = shuffled_class_indices[code]
                
                for i in range(start, end):
                    batch_part.append(class_idx_list[i % len(class_idx_list)])
                
                batch_indices.extend(batch_part)
                pointers[code] = end
                
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.length

# sampler.py (Corrected Logic)

import torch
import numpy as np
import logging
from torch.utils.data import Sampler
from typing import Iterator, List, Dict
import math

class HierarchicalBalancedBatchSampler(Sampler):
    """
    계층적 균형 샘플러:
    1. 최상위: Live(50%) vs Fake(50%)
    2. Fake 그룹 내: 모든 공격 유형(원본 코드 + Pseudo 코드)을 균등하게 샘플링
    """
    PSEUDO_ATTACK_CODES = [
        '1_0_0', '1_0_1', '1_0_2', '1_1_0', '1_1_1', '1_1_2',
        '2_0_0', '2_0_1', '2_0_2', '2_1_0', '2_1_1',
        '2_2_0', '2_2_1', '2_2_2'
    ]

    def __init__(self, dataset: 'FASDatasetWithInstruction', batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        if self.batch_size % 2 != 0:
            raise ValueError(f"Batch size ({self.batch_size}) must be even for 50:50 Live/Fake split.")
        
        # --- 1. 데이터셋 스캔 및 그룹화 ---
        self.live_indices = []
        self.real_fake_groups: Dict[str, List[int]] = {}

        for i, info in enumerate(self.dataset.samples_info):
            # '0_0_0' 코드를 가진 샘플을 Live로 간주
            if info['code'] == '0_0_0':
                self.live_indices.append(i)
            else: # 그 외 모든 코드는 원본 Fake
                code = info['code']
                if code not in self.real_fake_groups:
                    self.real_fake_groups[code] = []
                self.real_fake_groups[code].append(i)
        
        if not self.live_indices:
            raise ValueError("No live samples (code '0_0_0') found in the dataset for pseudo-attack generation.")

        # --- 2. Pseudo-Fake 그룹 생성 ---
        self.pseudo_fake_groups: Dict[str, List[int]] = {
            f"PSEUDO_{code}": self.live_indices for code in self.PSEUDO_ATTACK_CODES
        }
        
        # --- 3. 모든 Fake 그룹 통합 ---
        self.all_fake_groups = {**self.real_fake_groups, **self.pseudo_fake_groups}
        
        self.num_live_samples_per_batch = self.batch_size // 2
        self.num_fake_samples_per_batch = self.batch_size // 2
        self.num_fake_attack_types = len(self.all_fake_groups)
        
        # --- 4. 에포크 길이 계산 ---
        # 가장 작은 Fake 공격 유형 그룹의 크기를 기준으로 에포크 길이 결정
        min_fake_group_size = min(len(v) for v in self.all_fake_groups.values()) if self.all_fake_groups else 0
        
        num_fake_samples_per_epoch = min_fake_group_size * self.num_fake_attack_types
        self.num_samples_per_epoch = num_fake_samples_per_epoch * 2 # Live 샘플도 같은 수만큼 포함

        logging.info("HierarchicalBalancedBatchSampler Initialized")
        logging.info(f"Total Live (0_0_0) samples: {len(self.live_indices)}")
        logging.info(f"Total Real Fake attack types: {len(self.real_fake_groups)}")
        logging.info(f"Total Pseudo Fake attack types: {len(self.pseudo_fake_groups)}")
        logging.info(f"Total Fake attack types for sampling: {self.num_fake_attack_types}")
        logging.info(f"Batch Size: {self.batch_size} (Live: {self.num_live_samples_per_batch}, Fake: {self.num_fake_samples_per_batch})")
        logging.info(f"Total samples per epoch: {self.num_samples_per_epoch}")

    def __iter__(self) -> Iterator[List[tuple]]:
        live_pool = np.random.permutation(self.live_indices).tolist()
        fake_pools = {k: np.random.permutation(v).tolist() for k, v in self.all_fake_groups.items()}
        fake_group_keys = list(self.all_fake_groups.keys())

        num_batches = math.ceil(self.num_samples_per_epoch / self.batch_size)
        
        for _ in range(num_batches):
            batch_instructions = []
            
            # --- Live 샘플 샘플링 ---
            for _ in range(self.num_live_samples_per_batch):
                if not live_pool:
                    live_pool.extend(np.random.permutation(self.live_indices).tolist())
                batch_instructions.append((live_pool.pop(), 'LIVE'))
            
            # --- Fake 샘플 샘플링 (모든 공격 유형에서 순환하며 균등하게) ---
            for i in range(self.num_fake_samples_per_batch):
                group_key = fake_group_keys[i % self.num_fake_attack_types]
                
                if not fake_pools[group_key]:
                    fake_pools[group_key].extend(np.random.permutation(self.all_fake_groups[group_key]).tolist())
                
                original_idx = fake_pools[group_key].pop()
                
                if group_key.startswith("PSEUDO_"):
                    attack_code_to_apply = group_key.replace("PSEUDO_", "")
                else: # 원본 Fake는 그대로 사용
                    attack_code_to_apply = 'REAL_FAKE'
                
                batch_instructions.append((original_idx, attack_code_to_apply))

            random.shuffle(batch_instructions)
            yield batch_instructions

    def __len__(self) -> int:
        return math.ceil(self.num_samples_per_epoch / self.batch_size)
    

class FinalContrastiveSampler(Sampler):
    def __init__(self, dataset: 'FASFinalDataset', batch_k: int, batch_p: int):
        self.dataset = dataset
        self.batch_k = batch_k # 배치에 포함할 클래스(공격 유형) 수
        self.batch_p = batch_p # 클래스당 샘플 수
        self.batch_size = self.batch_k * self.batch_p

        # --- 데이터 그룹화 ---
        self.live_indices = []
        self.real_fake_indices_by_id = defaultdict(list)
        for i, info in enumerate(self.dataset.samples_info):
            # Live 샘플 (attack_id == 0)
            if info['attack_id'] == 0:
                self.live_indices.append(i)
            # Real Fake 샘플
            else:
                self.real_fake_indices_by_id[info['attack_id']].append(i)
        
        # 샘플링 가능한 Fake 공격 ID 리스트 (P개 이상 샘플이 있는 경우)
        self.fake_attack_ids = [
            aid for aid, indices in self.real_fake_indices_by_id.items() 
            if len(indices) >= self.batch_p // 2 # Pseudo-Fake와 섞기 위해 최소 P/2개 필요
        ]
        
        # K-1개의 Fake 클래스를 뽑아야 하므로, 유효한 Fake 클래스가 충분한지 확인
        if len(self.fake_attack_ids) < self.batch_k - 1:
            raise ValueError(f"Not enough fake attack types with sufficient samples for K={self.batch_k}")
        
        self.num_batches = len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_instructions = []
            
            # --- 1. K개의 클래스 선택: Live 1개 + Fake (K-1)개 ---
            selected_fake_ids = random.sample(self.fake_attack_ids, self.batch_k - 1)
            selected_class_ids = [0] + selected_fake_ids # 0은 Live의 ID

            # --- 2. 각 클래스에서 P개의 샘플 샘플링 ---
            for class_id in selected_class_ids:
                if class_id == 0: # Live 클래스
                    indices = np.random.choice(self.live_indices, self.batch_p * (self.batch_k-1), replace=True)
                    for idx in indices:
                        batch_instructions.append((idx, 'LIVE', class_id))
                elif class_id in [7, 8, 9]: # Pseudo Fake 클래스
                    for _ in range(self.batch_p):
                        # 50% 확률로 Real Fake 또는 Pseudo Fake 선택
                        if random.random() < 0.5 and self.real_fake_indices_by_id[class_id]:
                            idx = random.choice(self.real_fake_indices_by_id[class_id])
                            batch_instructions.append((idx, 'REAL_FAKE', class_id))
                        else:
                            idx = random.choice(self.live_indices)
                            batch_instructions.append((idx, 'PSEUDO_FAKE', class_id))
                else: # Fake 클래스
                    for _ in range(self.batch_p):
                        # 50% 확률로 Real Fake 또는 Pseudo Fake 선택
                        if random.random() < 0.8 and self.real_fake_indices_by_id[class_id]:
                            idx = random.choice(self.real_fake_indices_by_id[class_id])
                            batch_instructions.append((idx, 'REAL_FAKE', class_id))
                        else:
                            idx = random.choice(self.live_indices)
                            batch_instructions.append((idx, 'PSEUDO_FAKE', class_id))
            
            random.shuffle(batch_instructions)
            yield batch_instructions

    def __len__(self):
        return self.num_batches