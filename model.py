# model.py (TypeError fixed and Adapter logic improved)

import torch
import torch.nn as nn
import open_clip
from types import MethodType # 메서드를 동적으로 바인딩하기 위해 추가
from torch.nn import functional as F
# --- Adapter Module Definition (변경 없음) ---
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.non_linear(x)
        x = self.up_project(x)
        return x + residual

class DFFModule(nn.Module):
    """
    Deep Frequency Filtering (DFF) Module
    
    This module applies frequency domain filtering to intermediate features
    using FFT and learnable attention mechanism.
    input shape : (B, K, D) Batch size, K sequence length, D feature dimension
    output shape: (B, K, D) Filtered features in the same shape as input
    Input shape: torch.Size([128, 257, 1024]))
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(DFFModule, self).__init__()
        self.in_channels = in_channels
        
        # Embedding layer for frequency features
        self.magnitude_embedding = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.ReLU(inplace=True)
        )
        # Spatial attention module for frequency filtering
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        """
        Args:
            x: Input feature tensor of shape (B, K, D)
        
        Returns:
            filtered_x: Filtered feature tensor of shape (B, K, D)
        """
        B, K, D = x.shape
        
        # 1. FFT
        x_fft = torch.fft.fft(x, dim=-1)
        
        # 2. 크기(Magnitude)와 위상(Phase) 분리
        magnitude = x_fft.abs()
        phase = x_fft.angle()
        
        # 3. 크기를 (B*K, D) 형태로 펼쳐서 임베딩 및 어텐션 적용
        mag_flat = magnitude.view(B * K, -1)
        mag_embedded = self.magnitude_embedding(mag_flat)
        attention_mask = self.attention(mag_embedded)
        
        # 4. 필터링된 크기 계산
        mag_filtered = mag_flat * attention_mask
        mag_filtered = mag_filtered.view(B, K, -1)
        
        # 5. 필터링된 크기와 원본 위상으로 복소수 재구성
        real_part = mag_filtered * torch.cos(phase)
        imag_part = mag_filtered * torch.sin(phase)
        x_fft_filtered = torch.complex(real_part, imag_part)
        
        # 6. IFFT
        x_filtered = torch.fft.ifft(x_fft_filtered, n=D, dim=-1).real
        
        return x_filtered # Residual Connection 추가

class FrequencyAdapter(nn.Module):
    # Utilizing Deep Frequency Filtering 
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.dff = DFFModule(input_dim, reduction_ratio=bottleneck_dim)
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.non_linear = nn.GELU()
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        # self.dff_adapter = nn.Linear(input_dim, input_dim)  # Linear layer for DFF output
    def forward(self, x):
        residual = x
        rgb_out = self.dff(x)  # Apply DFF module
        rgb_out = self.down_project(rgb_out)
        rgb_out = self.non_linear(rgb_out)
        rgb_out = self.up_project(rgb_out)
        return rgb_out + residual  # Residual connection
        
        
        
        

class ClipFas(nn.Module):
    def __init__(self, num_classes=2, model_name='ViT-L-14', pretrained='laion2b_s32b_b82k',
                 finetune_strategy='adapter', bottleneck_dim=64):
        super(ClipFas, self).__init__()
        print(f"Loading CLIP model: {model_name} with pretrained weights: {pretrained}")
        if finetune_strategy == 'Swin':
            import timm
            tmp_model = timm.create_model('swin_large_patch4_window7_224', pretrained=True)
            embed_dim = tmp_model.head.fc.in_features
            self.visual_encoder = tmp_model # Swin Transformer의 경우, visual_encoder는 모델 자체가 됨
            self.visual_encoder.float()
            # Swin Transformer의 경우, head를 제거하고 AdaptiveAvgPool2d를 사용
            self.visual_encoder.head.fc = nn.Identity()  # head를 Identity로 설정하여 제거
            self.fc = nn.Linear(embed_dim, num_classes)
                
        else:
            self.clip, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=torch.device('cpu')
            )

            self.visual_encoder = self.clip.visual
            embed_dim = 768
            self.fc = nn.Linear(embed_dim, num_classes)
            self.apply_finetuning_strategy(finetune_strategy, bottleneck_dim)
    # 모델 device setting 시, self.visual_encoder와 self.fc를 동일한 device로 설정
    def to(self, device):
        super(ClipFas, self).to(device)
        self.visual_encoder.to(device)
        self.fc.to(device)
        return self

    # model.float 시, self.visual_encoder와 self.fc를 동일한 dtype으로 설정
    def float(self):
        super(ClipFas, self).float()
        self.visual_encoder.float()
        self.fc.float()

    # half 시, self.visual_encoder와 self.fc를 동일한 dtype으로 설정
    def half(self):
        super(ClipFas, self).half()
        self.visual_encoder.half()
        self.fc.half()
    def eval(self):
        super(ClipFas, self).eval()
        self.visual_encoder.eval()
        self.fc.eval()
    def train(self, mode=True):
        super(ClipFas, self).train(mode)
        self.visual_encoder.train(mode)
        self.fc.train(mode)
    # --- Fine-tuning Strategy 적용 ---
    def apply_finetuning_strategy(self, strategy, bottleneck_dim):
        print(f"Applying fine-tuning strategy: '{strategy}'")
        
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        if strategy == 'adapter':
            for i, block in enumerate(self.visual_encoder.transformer.resblocks):
                # 각 블록에 어댑터 모듈 추가
                block.adapter = Adapter(block.attn.out_proj.out_features, bottleneck_dim)
                
                # --- SOLUTION: Modify the forward method correctly ---
                # 원본 forward 메서드를 저장
                original_forward = block.forward

                # 새로운 forward 메서드 정의 (self와 attn_mask 인자 추가)
                def new_forward(self, x, attn_mask=None):
                    # 먼저 원본 블록의 forward를 그대로 실행
                    x = original_forward(x, attn_mask=attn_mask)
                    # 그 결과에 어댑터를 적용
                    x = self.adapter(x)
                    return x
                # MethodType을 사용하여 new_forward를 인스턴스 메서드로 바인딩
                block.forward = MethodType(new_forward, block)

            # 학습할 파라미터 설정
            for name, param in self.visual_encoder.named_parameters():
                if 'adapter' in name or 'ln_post' in name:
                    param.requires_grad = True
        elif strategy == 'frequency_adapter':
            for i, block in enumerate(self.visual_encoder.transformer.resblocks):
                # 각 블록에 FrequencyAdapter 모듈 추가
                block.frequency_adapter = FrequencyAdapter(block.attn.out_proj.out_features, bottleneck_dim)
                # --- SOLUTION: Modify the forward method correctly ---
                # 원본 forward 메서드를 저장
                original_forward = block.forward
                # 새로운 forward 메서드 정의 (self와 attn_mask 인자 추가)
                def new_forward(self, x, attn_mask=None):
                    # 먼저 원본 블록의 forward를 그대로 실행
                    x = original_forward(x, attn_mask=attn_mask)
                    # 그 결과에 FrequencyAdapter를 적용
                    x = self.frequency_adapter(x)
                    return x
                # MethodType을 사용하여 new_forward를 인스턴스 메서드로 바인딩
                block.forward = MethodType(new_forward, block)
            # 학습할 파라미터 설정
            for name, param in self.visual_encoder.named_parameters():
                if 'frequency_adapter' in name or 'ln_post' in name:
                    param.requires_grad = True
        elif strategy == 'partial':
            # ... (이전과 동일)
            for i in range(20, 24):
                for param in self.visual_encoder.transformer.resblocks[i].parameters():
                    param.requires_grad = True
            for param in self.visual_encoder.ln_post.parameters():
                param.requires_grad = True

        elif strategy == 'linear_probe':
            # ... (이전과 동일)
            pass

        elif strategy == 'full':
            # ... (이전과 동일)
            for param in self.visual_encoder.parameters():
                param.requires_grad = True
        
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
            
        for param in self.fc.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")


    def forward(self, x, return_features=False):

        features = self.visual_encoder(x)        
        logits = self.fc(features)
        
        if return_features:
            # 정규화된 특징 벡터와 로짓을 함께 반환
            return logits, F.normalize(features, dim=1)
        else:
            return logits
