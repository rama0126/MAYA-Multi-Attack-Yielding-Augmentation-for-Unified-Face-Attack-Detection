# augmentation.py (Final Integrated and Refined Version)

import cv2
import numpy as np
import random
import joblib
import dlib
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import albumentations as A

import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='albumentations')
import os

# ===================================================================
# --- 1. Helper Functions (dlib and Pre-computed Mask) ---
# ===================================================================

# --- Dlib 초기화 ---
try:
    # dlib 모델 파일 경로를 정확히 지정해야 합니다.
    predictor_path = "/workspace/FAS_ICCV/models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    DLIB_AVAILABLE = True
except Exception as e:
    print(f"Warning: dlib model not found at '{predictor_path}' or failed to load. Landmark-based augmentations will be disabled. Error: {e}")
    DLIB_AVAILABLE = False

def get_landmarks(image_np):
    """dlib을 사용하여 얼굴 랜드마크를 실시간으로 추출합니다."""
    if not DLIB_AVAILABLE or image_np is None: return None
    img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 1)
    if len(faces) == 0: return None
    shape = predictor(img_gray, faces[0])
    return np.array([[p.x, p.y] for p in shape.parts()])

def load_mask_and_get_parts(mask_path):
    """미리 계산된 마스크 파일(.pkl)을 로드하고, 각 얼굴 부위의 좌표(contours)를 반환합니다."""
    try:
        mask = joblib.load(mask_path)
    except Exception:
        return None

    part_indices = {
        'face': list(range(1, 14)), 'l_eye': [4], 'r_eye': [5], 'eyes': [4, 5],
        'l_brow': [2], 'r_brow': [3], 'brows': [2, 3], 'nose': [10],
        'mouth': [11, 12, 13]
    }
    
    parts = {}
    for part_name, indices in part_indices.items():
        binary_mask = np.isin(mask, indices).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            all_points = np.concatenate(contours, axis=0)
            if all_points.size > 0:
                parts[part_name] = all_points
    return parts if parts else None

class MoirePattern:
    """모아레 패턴 시뮬레이션 클래스."""
    def __call__(self, img):
        """
        Generates a moiré pattern using sinusoidal interference and applies it to the image.
        Args:
            img (ndarray or PIL Image): Input image.
            intensity (float): Strength of the moiré effect.
        Returns:
            ndarray: Image with moiré pattern applied.
        """
        h, w = img.shape[:2]
        intensity= random.uniform(0.1, 0.4)  # Random intensity between 0.1 and 0.5
        # Convert to array
        if isinstance(img, Image.Image):
            arr = np.array(img).astype(np.float32)
        else:
            arr = img.astype(np.float32)
        h, w = arr.shape[:2]
        # Create meshgrid
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        # First sine wave
        freq1 = random.uniform(0.1, 0.5)
        angle1 = random.uniform(0, np.pi)
        moire1 = np.sin(freq1 * (X * np.cos(angle1) + Y * np.sin(angle1)))
        # Second sine wave
        freq2 = random.uniform(0.1, 0.5)
        angle2 = random.uniform(0, np.pi)
        if abs(angle1 - angle2) < 0.2:
            angle2 = (angle1 + np.pi/2) % np.pi
        moire2 = np.sin(freq2 * (X * np.cos(angle2) + Y * np.sin(angle2)))
        # Combine
        combined = (moire1 + moire2) / 2.0  # range [-1,1]
        # Apply per channel
        noise_val = intensity * 127.5
        for c in range(3):
            arr[:,:,c] += combined * random.uniform(0.5 * noise_val, noise_val)
        # Clip and convert back
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        del moire1, moire2, X, Y, combined, noise_val, x, y
        return arr
    
# ===================================================================
# --- 2. Physical Attack Simulations (고도화 및 통합) ---
# ===================================================================
# 1. Print Attack (simulate_print)
# ShiftScaleRotate + ChannelShuffle

# 무작위 채널 분리·합성으로 잉크 번짐과 색상 오차를 흉내 냄.

# 하프톤(Dot) 패턴

# 반복되는 작은 원을 그레이스케일 밝기에 비례해 찍어, 신문·잡지 인쇄 특유의 도트 그리드(하프톤) 효과 재현.

# 종이 질감 노이즈

# 랜덤 노이즈를 더해 종이 표면의 거칠고 톡톡 튀는 질감을 모방.

# MoirePattern

# 인쇄된 이미지가 촬영·스캔될 때 생기는 모아레 패턴을 시뮬레이션.

# GaussianBlur + ImageCompression

# 약한 블러로 인쇄·스캔 후 선명도 저하를,

# JPEG 압축으로 인쇄물 스캔 특유의 블록 아티팩트와 압축 노이즈를 함께 재현.

PRINT_AUG = A.Compose([A.ChannelShuffle(p=0.3), 
                     A.ColorJitter(hue=0.1, saturation=0.1, brightness=0.1, contrast=0.1, p=0.5)])
PRINT_GEO =A.Compose([
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.7),
        A.ElasticTransform(alpha=1.0, sigma=50,  p=0.5),
        A.Affine(translate_percent=0.05, rotate=(-5,5), scale=(0.95,1.05), p=0.5),

        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5)
        ])
PRINT_BLUR = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    A.OneOf([
            A.ImageCompression(quality_lower=40, quality_upper=90, compression_type="jpeg", p=1.0),
            A.ImageCompression(quality_lower=40, quality_upper=90, compression_type="webp", p=1.0),
        ], p=0.8),
])
MOARAE_PATTERN = MoirePattern()
def simulate_print(img_np):
    """1_0_0: Print Attack. 인쇄물의 특성(색 번짐, 하프톤, 질감, 모아레)을 시뮬레이션."""
    h, w, _ = img_np.shape
    original_img = img_np.copy()
    img_np = PRINT_BLUR(image=img_np)['image']
    # 미세한 색상 채널 분리
    r, g, b = cv2.split(img_np)
    r, g = PRINT_AUG(image=r)['image'], PRINT_AUG(image=g)['image']
    img_np = cv2.merge([r, g, b])
    del r, g, b
    # 하프톤 패턴 모방
    if random.random() < 0.2:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        dot_pattern = np.zeros_like(gray, dtype=np.float32)
        for size in [2, 3]:
            for i in range(0, h, size * 2):
                for j in range(0, w, size * 2):
                    radius = int(gray[i, j] / 255.0 * size * 0.8)
                    if radius > 0: cv2.circle(dot_pattern, (j, i), radius, 1, -1)
        dot_pattern_color = cv2.cvtColor((dot_pattern * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_np = cv2.addWeighted(img_np, 0.85, dot_pattern_color, 0.15, 0)
        del gray, dot_pattern, dot_pattern_color
    # 종이 질감 추가
    noise = cv2.merge([np.random.uniform(0, 1, (h, w, 1)) * 15] * 3)
    img_np = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    del noise
    # 모아레 패턴 추가
    if random.random() < 0.5:
        # img_np = MoirePattern()(img_np)
        alpha = random.uniform(0.05, 0.4)  # 모아레 강도
        img_np = img_np * (1 - alpha) + MOARAE_PATTERN(img_np) * alpha
        
    if random.random() < 0.8:
        # 원본 이미지에 작게 geometric 변형 적용한 이미지를 합성
        # img_np = geo_transform(image=img_np)['image']
        img_np = PRINT_GEO(image=img_np)['image']
        patch_scale_range=(0.8, 0.99)
        # 패치 크기 및 위치
        scale = random.uniform(*patch_scale_range)
        ph, pw = int(h * scale), int(w * scale)
        top = random.randint(0, h - ph)
        left = random.randint(0, w - pw)
        img_np = cv2.resize(img_np, (pw, ph), interpolation=cv2.INTER_LINEAR)
        spoofed = original_img.copy()
        spoofed[top:top+ph, left:left+pw] = img_np
        img_np = spoofed
        del spoofed
    # img_np = A.GaussianBlur(blur_limit=(3, 5), p=0.5)(image=img_np)['image']
    # img_np = A.ImageCompression(quality_lower=50, quality_upper=90, p=0.8)(image=img_np)['image']

    del original_img
    return img_np
# 2. Replay Attack (simulate_replay)
# 픽셀 그리드 효과 (Pixelation)

# 소형 해상도→원본 크기 보간 처리로, 화면 녹화·재생 시 생기는 픽셀 러닝(모자이크) 현상을 흉내.

# 랜덤 SunFlare (렌즈 플레어)

# 스마트폰·카메라로 화면을 촬영할 때 들어오는 렌즈 반사광을 시뮬레이션.

# 시야각에 따른 색편차(Tint)

# 가우시안 블러된 마스크로 화면 중심부와 모서리의 색온도 차이를 재현(파란빛·붉은빛 틴트).

# MoirePattern

# 디스플레이 픽셀 구조와 카메라 센서 간 모아레 재현.
REPLAY_GEO =A.Compose([
        A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.7),
        # A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=10, p=0.5),
        A.ElasticTransform(alpha=1.0, sigma=50, p=0.5),
        A.Affine(translate_percent=0.05, rotate=(-5,5), scale=(0.95,1.05), p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5)
        ])
# REPLAY_SUNFLARE = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=150, p=0.2)
REPLAY_SUNFLARE = A.RandomSunFlare(
  flare_roi=(0,0,1,1),
  num_flare_circles_lower=3, num_flare_circles_upper=5,
  src_radius=150,
  p=0.2
)
REPLAY_AUG = A.Compose([
    A.ChannelShuffle(p=0.3), 
    A.ColorJitter(hue=0.1, saturation=0.1, brightness=0.1, contrast=0.1, p=0.5),
    A.OneOf([
            A.ImageCompression(quality_lower=30, quality_upper=80, compression_type="jpeg", p=1.0),
            A.ImageCompression(quality_lower=30, quality_upper=80, compression_type="webp", p=1.0),
        ], p=0.8),
    A.OneOf([
            A.GaussNoise(var_limit=(10, 30), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.5)]
    , p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
])
def simulate_replay(img_np):
    """1_0_1: Replay Attack. 스크린의 픽셀 그리드, 반사, 색 왜곡, 모아레를 시뮬레이션."""
    h, w, _ = img_np.shape
    
    # 스크린 픽셀 그리드 효과
   
    grid_size = random.choice([2, 4, 8])
    small = cv2.resize(img_np, (w // grid_size, h // grid_size), interpolation=cv2.INTER_NEAREST)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img_np = cv2.addWeighted(img_np, 0.6, pixelated, 0.4, 0)
    del small, pixelated, grid_size
    if random.random() < 0.7:
        img_np = REPLAY_GEO(image=img_np)['image']
        
    img_np = REPLAY_AUG(image=img_np)['image']
    # 렌즈 플레어 및 화면 반사
    # img_np = A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=1, num_flare_circles_upper=3, src_radius=150, p=0.2)(image=img_np)['image']
    img_np = REPLAY_SUNFLARE(image=img_np)['image']

    # 시야각에 따른 색상 변화
    if random.random() < 0.5:
        color = random.choice([(0, 0, 50), (50, 0, 0), (0, 50, 0), (50, 50, 0), (0, 50, 50), (50, 0, 50)])
        mask = cv2.GaussianBlur(cv2.erode(np.full((h, w, 1), 255, dtype=np.uint8), np.ones((15,15))), (51, 51), 0)[..., np.newaxis] / 255.0
        tint_layer = np.full_like(img_np, color)
        img_np = np.clip(img_np * (1-mask*0.2) + tint_layer * (mask*0.2), 0, 255).astype(np.uint8)
        del mask, tint_layer, color
    # 모아레 패턴 추가
    if random.random() < 0.4:
        img_np = MOARAE_PATTERN(img_np)
    else :
        alpha = random.uniform(0.05, 0.5)  # 모아레 강도
        img_np = img_np * (1 - alpha) + MOARAE_PATTERN(img_np) * alpha
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    # 최종 이미지 반환
    return img_np

import cv2
import numpy as np
import albumentations as A
import random

def scale_contour(contour: np.ndarray, scale: float) -> np.ndarray:
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return contour
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    scaled = []
    del M
    for p in contour.squeeze():
        x, y = p
        x_new = (x - cx) * scale + cx
        y_new = (y - cy) * scale + cy
        scaled.append([[int(x_new), int(y_new)]])
    return np.array(scaled, dtype=np.int32)

def perturb_contour(contour, jitter=2,jitter_nodes=5):
    """
    Randomly perturbs a fixed number of outer points of a convex hull contour by up to ±jitter pixels.
    Only jitter_nodes points on the hull are moved; others remain unchanged.

    Args:
        contour (ndarray): Input contour array (N,1,2).
        jitter (float): Max pixel offset for jitter.
        jitter_nodes (int): Exact number of hull nodes to perturb.
    Returns:
        ndarray: Perturbed contour of same shape.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    # compute hull indices on pts
    hull = cv2.convexHull(pts, returnPoints=False)
    hull_indices = hull.flatten().tolist()
    num = len(hull_indices)
    k = min(jitter_nodes, num)
    selected = random.sample(hull_indices, k)
    # prepare jittered points list
    jittered_pts = pts.copy()
    for idx in selected:
        jittered_pts[idx] = pts[idx] + np.random.uniform(-jitter, jitter, 2)
    # reconstruct contour shape
    perturbed = jittered_pts.reshape(contour.shape).astype(np.int32)
    return smooth_contour(perturbed)

def smooth_contour(contour, kernel_size=5):
    """
    Smooths a contour by averaging each point with its neighbors.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)
    half = kernel_size // 2
    smoothed = []
    for i in range(n):
        idxs = [(i + j) % n for j in range(-half, half + 1)]
        neighbors = np.array([pts[idx] for idx in idxs])
        avg = neighbors.mean(axis=0)
        smoothed.append(avg)
    smoothed = np.array(smoothed, dtype=np.int32).reshape(contour.shape)
    
    return smoothed

# 3. Cutout Attack (simulate_cutouts)
# Scaled Contour + Feather Mask

# face 컨투어를 0.9~1.1 배 스케일로 변형해, 마스크의 위치·크기 오차를 흉내.

# (101×101) 가우시안 블러로 경계부를 부드럽게 페더링하여, 자연스럽게 얼굴 일부만 가려진 효과.

# 강한 그림자(Shadow) 블렌딩

# 가려진 영역을 80–95% 검은색으로 채워, 실제 컷아웃(구멍 뚫린 마스크)처럼 시각적 차단 효과를 극대화.

# Edge Highlight

# 컨투어 가장자리를 두껍게 드로잉 → 31×31 블러 → 50% 밝기 오버레이 처리해, 마스크 홀 경계가 살짝 떠오르는 듯한 하이라이트 연출.
CUTOUT_AUG =A.Compose([A.ChannelShuffle(p=0.3), 
                     A.ColorJitter(hue=0.1, saturation=0.1, brightness=0.1, contrast=0.1, p=0.5)])
CUTOUT_BLUR = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    A.OneOf([
            A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="jpeg", p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="webp", p=1.0),
        ], p=0.5),
])
def simulate_cutouts(img_np, face_parts):
    """
    Subtle Cutout Attack:
    # phase 1: Apply subtle printing attack
    # phase 2: Apply cutout attack on face parts
    ## phase 1 - Apply subtle printing artifact 
    ## phase 2-1: make mask from face parts
    ## phase 2-2: cutout face parts on the mask
    ## phase 2-3: overwrite cutout face parts with original image
    """
    # pha
    h, w = img_np.shape[:2]
    original_img = img_np.copy()
    img_aug = img_np.copy()
    # 1) Create mask from scaled face contours
    r, g, b = cv2.split(img_aug)
    r, g = CUTOUT_AUG(image=r)['image'], CUTOUT_AUG(image=g)['image']
    img_aug = cv2.merge([r, g, b])
    del r, g, b
    # 하프톤 패턴 모방
    if random.random() < 0.2:
        gray = cv2.cvtColor(img_aug, cv2.COLOR_RGB2GRAY)
        dot_pattern = np.zeros_like(gray, dtype=np.float32)
        for size in [2, 3]:
            for i in range(0, h, size * 2):
                for j in range(0, w, size * 2):
                    radius = int(gray[i, j] / 255.0 * size * 0.8)
                    if radius > 0: cv2.circle(dot_pattern, (j, i), radius, 1, -1)
        dot_pattern_color = cv2.cvtColor((dot_pattern * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img_aug = cv2.addWeighted(img_aug, 0.85, dot_pattern_color, 0.15, 0)
        del gray, dot_pattern, dot_pattern_color
    # 종이 질감 추가
    if random.random() < 0.5:
        noise = cv2.merge([np.random.uniform(0, 1, (h, w, 1)) * 15] * 3)
        img_aug = np.clip(img_aug.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        del noise
    # 모아레 패턴 추가
    if random.random() < 0.3:
        img_aug = MOARAE_PATTERN(img_aug)
    
    img_aug = CUTOUT_BLUR(image=img_aug)['image']
    # Phase 2
    
    # 2) Create mask from face parts
    mask = np.zeros((h, w), dtype=np.uint8)
    if 'face' in face_parts:
        # 얼굴부분을 기준으로 margin random margin 주고 사각형 마스크 생성
        face_contour = face_parts['face']
        x, y, w, h = cv2.boundingRect(face_contour)
        margin_ratio = random.uniform(0.05, 0.15)  # 5% ~ 15% 사이의 랜덤 마진
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + w + margin_x)
        y2 = min(h, y + h + margin_y)
        mask[y1:y2, x1:x2] = 1
    if 'nose' in face_parts:
        if random.random() < 0.3:
            # 코 부분은 제거
            nose_contour = face_parts['nose']
            # 코 부분의 윤곽선의 크기를 랜덤하게 조정
            scale = random.uniform(0.8, 1.0)
            node_contour = random.randint(3,15)
            nose_contour = scale_contour(nose_contour, scale)
            # 윤곽선도 약간의 지터를 추가
            # jitter 크기는 contour의 크기에 비례하여 조정
            jitter = int(nose_contour.shape[0] * 0.1)  # contour의 2% 크기
            nose_contour = perturb_contour(nose_contour, jitter=jitter, jitter_nodes=node_contour)
            cv2.fillPoly(mask, [nose_contour], 0)
    if 'mouth' in face_parts:
        if random.random() < 0.7:
            # 입술 부분은 제거
            mouth_contour = face_parts['mouth']
            # 입술 부분의 윤곽선의 크기를 랜덤하게 조정
            scale = random.uniform(0.6, 1.0)
            nose_contour = random.randint(3,15)
            mouth_contour = scale_contour(mouth_contour, scale)
            # 윤곽선도 약간의 지터를 추가
            # jitter 크기는 contour의 크기에 비례하여 조정
            jitter = int(mouth_contour.shape[0] * 0.1)
            mouth_contour = perturb_contour(mouth_contour, jitter=jitter, jitter_nodes=nose_contour)
            # 입술 부분의 윤곽선을 제거
            cv2.fillPoly(mask, [mouth_contour], 0)
    if 'l_eye' in face_parts and 'r_eye' in face_parts:
        if random.random() < 0.7:
            # 눈 부분은 제거
            l_eye_contour = face_parts['l_eye']
            r_eye_contour = face_parts['r_eye']
            # 눈 부분의 윤곽선의 크기를 랜덤하게 조정
            scale = random.uniform(0.8, 1.0)
            node_contour = random.randint(3,15)
            l_eye_contour = scale_contour(l_eye_contour, scale)
            r_eye_contour = scale_contour(r_eye_contour, scale)
            # 윤곽선도 약간의 지터를 추가
            jitter = int(l_eye_contour.shape[0] * 0.1)
            l_eye_contour = perturb_contour(l_eye_contour, jitter=jitter, jitter_nodes=node_contour)
            r_eye_contour = perturb_contour(r_eye_contour, jitter=jitter, jitter_nodes=node_contour)
            # 눈 부분의 윤곽선을 제거
            cv2.fillPoly(mask, [l_eye_contour], 0)
            cv2.fillPoly(mask, [r_eye_contour], 0)
    if 'l_brow' in face_parts and 'r_brow' in face_parts:
        if random.random() < 0.1:
            # 눈썹 부분은 제거
            l_brow_contour = face_parts['l_brow']
            r_brow_contour = face_parts['r_brow']
            # 눈썹 부분의 윤곽선의 크기를 랜덤하게 조정
            scale = random.uniform(0.8, 1.0)
            node_contour = random.randint(3,15)
            l_brow_contour = scale_contour(l_brow_contour, scale)
            r_brow_contour = scale_contour(r_brow_contour, scale)
            # 윤곽선도 약간의 지터를 추가
            jitter = int(l_brow_contour.shape[0] * 0.1)
            l_brow_contour = perturb_contour(l_brow_contour, jitter=jitter, jitter_nodes=node_contour)
            r_brow_contour = perturb_contour(r_brow_contour, jitter=jitter, jitter_nodes=node_contour)
            # 눈썹 부분의 윤곽선을 제거
            cv2.fillPoly(mask, [l_brow_contour], 0)
            cv2.fillPoly(mask, [r_brow_contour], 0)
    # img_aug 에 geometric augmentation 적용
    # rotation, scaling, translation, perspective, shear
    # 적용할 때, mask도 함께 적용
    if random.random() < 0.5:
        # 회전
        angle = random.uniform(-10, 10)
        h, w = img_aug.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_aug = cv2.warpAffine(img_aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        del angle,center, M
    if random.random() < 0.5:
        # 스케일링
        scale = random.uniform(0.9, 1.1)
        h, w = img_aug.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        img_aug = cv2.warpAffine(img_aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        del scale,  M
    if random.random() < 0.5:
        # 평행이동
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        h, w = img_aug.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img_aug = cv2.warpAffine(img_aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        del tx, ty, M
    if random.random() < 0.5:
        # 원근 변환
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = pts1 + np.random.uniform(-10, 10, pts1.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_aug = cv2.warpPerspective(img_aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        del pts1, pts2, M
    if random.random() < 0.5:
        # 전단 변환
        shear = random.uniform(-0.1, 0.1)
        M = np.float32([[1, shear, 0], [shear, 1, 0]])
        img_aug = cv2.warpAffine(img_aug, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        del M, shear
    if mask.shape[0] != original_img.shape[0] or mask.shape[1] != original_img.shape[1]:
        # mask의 크기가 원본 이미지와 다를 경우, 원본 이미지 크기에 맞게 리사이즈
        mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_aug = cv2.resize(img_aug, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
    # 3) cutout face parts on the mask
    # mask를 0과 1로 이진화
    mask = (mask > 0.5).astype(np.uint8)  # 이진화
    # mask를 3채널로 확장
    mask = np.stack([mask] * 3, axis=-1)  # (h, w, 3) 형태로 확장
    # mask를 이용해서 mask 가 0인 부분은 원본 이미지로 대체
    img_aug[mask == 0] = original_img[mask == 0]
    if random.random() < 0.9:
        # original 부분에 shadow 추가
        shadow_mask = np.zeros_like(mask, dtype=np.uint8)
        shadow_mask[mask == 0] = 1
        # shadow_ratio = random.uniform(0.1, 0.6)  # 10% ~ 30% 사이의 랜덤 비율
        shadow_intensity = random.uniform(0.0,0.2)  # 그림자 강도
        shadow_color = np.random.uniform(0, 30, size=(1, 1, 3)).astype(np.uint8)  # 어두운 색상
        shadow = img_aug * shadow_intensity + shadow_color
        shadow[shadow < 0] = 0
        shadow[shadow > 255] = 255
        img_aug[shadow_mask == 1] = shadow[shadow_mask == 1]
        del shadow_mask, shadow_color, shadow_intensity, shadow
    # 5) 최종 이미지 반환
    img_np = img_aug.astype(np.uint8)
    # 잔여물 제거
    del original_img, img_aug, mask
    return img_np
def jitter_contour(contour: np.ndarray, jitter_range=(0.9, 1.1)) -> np.ndarray:
    """얼굴 윤곽점을 기준으로 각 점을 랜덤하게 스케일링하여 자유롭게 변형"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return contour
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    del M
    pts = contour.squeeze()
    jittered = []
    for (x, y) in pts:
        scale = random.uniform(*jitter_range)
        x_new = (x - cx) * scale + cx
        y_new = (y - cy) * scale + cy
        jittered.append([[int(x_new), int(y_new)]])
    return np.array(jittered, dtype=np.int32)

# 4. 3D Mask Attack (simulate_3d_mask)
# 여러 재질(transparent, plaster(Plastic), resin) 모드를 통해 실제 얼굴형 마스크의 물리적 성질을 재현했습니다.

# 4-1. Transparent Mode
# OpticalDistortion

# 플라스틱·유리 마스크를 통과한 광선 굴절(refraction) 초기 왜곡.

# 부분 Refraction

# mask_blur 기반 확률 영역에만 cv2.remap을 적용, 불규칙하게 굴절된 플라스틱 느낌 부여.

# Inner Reflection

# 가우시안 블러된 원본을 소량(over 5–15%) 섞어, 마스크 내부 면 반사광 효과 흉내.

# Fresnel Edge

# Canny→블러한 에지 마스크에 밝은 하이라이트를 50–80% 가중 블렌딩해, 투명체 가장자리에서 반사율이 높아지는 Fresnel 효과 재현.

# 4-2. Plastic Mode
# Tint & Transparency

# 얼굴 위에 연한 블루-그린 틴트를 70% 정도 블렌딩해 반투명 플라스틱 특유의 색조 부여.

# Specular Highlight

# Gaussian 블러된 원형 밝기를 투명도에 맞춰 오버레이해, 플라스틱 표면의 고광택 반짝임 반영.

# Edge Feathering

# 마스크 외곽부를 (101×101) 가우시안 블러해 얼굴이 자연스럽게 비치는 느낌 유지.

# 4-3. Resin Mode
# Warm Tint & SSS(Subsurface Scattering)

# 주홍빛-핑크빛 틴트를 얼굴에 블렌딩하고, Gaussian 블러로 내부 빛 확산 효과(SSS) 흉내.

# Specular Highlight Ring

# 랜덤 위치에 블러된 링 형태 반짝임 추가해, 레진의 두께감 있는 반짝임 강조.

# Matte × Gloss Mix

# 글로시와 매트를 60 : 40 비율로 섞어, 자연스러운 빛 흡수·반사 특성을 재현.

# Edge Feathering

# 마스크 외곽부를 mask_blur로 부드럽게 블렌딩하여, 레진 속 얼굴 윤곽이 은은히 비치는 효과.
def simulate_3d_mask(img_np, face_parts, mode='transparent'):
    """1_1_x: 얼굴 윤곽을 기반으로 하되 랜덤하게 유영하는 3D Mask Attack"""
    h, w = img_np.shape[:2]
    # 얼굴 윤곽 기반 컨투어 추출 및 랜덤 변형
    hull = cv2.convexHull(face_parts['face'])
    # 컨투어을 자유롭게 변형 (0.9~1.1 스케일)
    mask_contour = jitter_contour(hull, jitter_range=(0.9, 1.1))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [mask_contour], -1, 255, -1)
    # 경계 흐림
    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)

    alpha = (mask_blur / 255.0)[..., None] * 0.5
    
    if mode == 'transparent':
        # 1) 기본 왜곡
        
        return simulate_mask_trans(img_np, face_parts)
    elif mode == 'plaster':
        # Plastic mode: 얼굴 모양 마스크에 플라스틱 재질감 적용
        # 1) 틴트 색상 (연한 블루 그린 계열)
        tint = np.full_like(img_np, (180, 200, 220), dtype=np.uint8)
        # 2) 마스크 투명도+틴트 블렌딩
        plastic_overlay = (img_np.astype(np.float32) * (1 - alpha*0.7) +
                           tint.astype(np.float32) * (alpha*0.7))
        plastic_overlay = np.clip(plastic_overlay, 0, 255).astype(np.uint8)
        # 3) 간단한 하이라이트 (Specular)
        # 밝은 원형 하이라이트 생성
        highlight = np.zeros_like(img_np)
        cx, cy = np.unravel_index(np.argmax(mask), mask.shape)
        radius = int(min(h, w) * 0.15)
        cv2.circle(highlight, (cx, cy), radius, (255,255,255), -1)
        highlight = cv2.GaussianBlur(highlight, (101,101), 0)
        spec_strength = 0.2
        # result = cv2.addWeighted(plastic_overlay, 1.0, highlight, spec_strength * (mask_blur/255.0)[..., None], 0)
        result = (plastic_overlay.astype(np.float32) * (1 - spec_strength * (mask_blur/255.0)[..., None]) +
                  highlight.astype(np.float32) * (spec_strength * (mask_blur/255.0)[..., None]))
        # 4) 투명 가장자리 페더링
        edge_feather = cv2.GaussianBlur(mask.astype(np.uint8), (101,101), 0).astype(np.float32)/255.0
        result = (result.astype(np.float32) * edge_feather[..., None] +
                  img_np.astype(np.float32) * (1-edge_feather[..., None]))
        return np.clip(result, 0, 255).astype(np.uint8)



    elif mode == 'resin':
        # Resin mode: 반투명하고 광택 있는 레진 재질감
        # 1) 틴트 색상 (따듯한 주황 또는 핑크 계열)
        resin_tint = np.full_like(img_np, (230, 180, 170), dtype=np.uint8)
        # 2) 기본 베이스: 원본과 틴트의 블렌딩
        base_tint = (img_np.astype(np.float32) * (1 - alpha) + resin_tint.astype(np.float32) * alpha)
        # 3) subsurface scattering 흉내: 내부 퍼짐 효과
        if random.random() < 0.5:
            sss = cv2.GaussianBlur(base_tint.astype(np.uint8), (51, 51), sigmaX=30)
            sss_strength = 0.3
            base_sss = (base_tint * (1 - sss_strength) + sss.astype(np.float32) * sss_strength)
        else:
            base_sss = base_tint
        # 4) 광택 하이라이트 (specular)
        highlight = np.zeros_like(img_np)
        # highlight 위치 랜덤 링 형태
        angle = random.uniform(0, 2*np.pi)
        r = int(min(h, w) * 0.2)
        cx, cy = int(w/2 + r*np.cos(angle)), int(h/2 + r*np.sin(angle))
        cv2.circle(highlight, (cx, cy), int(r*0.6), (255,255,255), -1)
        highlight = cv2.GaussianBlur(highlight, (101, 101), 0)
        spec_strength = 0.25
        # base_spec = cv2.addWeighted(base_sss.astype(np.uint8), 1.0, highlight, spec_strength * (mask_blur/255.0)[..., None], 0)
        base_spec = (base_sss.astype(np.float32) * (1 - spec_strength * (mask_blur/255.0)[..., None]) +
                     highlight.astype(np.float32) * (spec_strength * (mask_blur/255.0)[..., None]))
        # 5) Matte와 Gloss 믹스: Glossy 부분과 매트한 부분 섞기
        matte_strength = 0.6
        matte = cv2.GaussianBlur(base_spec, (15, 15), 0)
        resin_final = (base_spec.astype(np.float32) * matte_strength + matte.astype(np.float32) * (1 - matte_strength))
        # 6) 가장자리 페더링: resin 영역외부는 원본
        feather = (mask_blur/255.0)[..., None]
        result = resin_final * feather + img_np.astype(np.float32) * (1 - feather)
        return np.clip(result, 0, 255).astype(np.uint8)

    return img_np



# ===================================================================
# --- 3. Digital Attack Simulations (고도화 및 통합) ---
# ===================================================================
# 요약
# 부위 단위 마스킹 + Soft Blur
# → 편집 영역 경계가 부드럽게 섞여 실제 GAN 편집처럼 보임

# ColorJitter
# → 눈·입술·코 등 특정 부위의 색·명암·채도 변화를 간단히 모사

# 랜덤 부위 선택
# → 다양한 종류의 속성 편집 공격(색상, 크기, 형태 변경 등)을 폭넓게 학습
ATTRIBUTE_EDIT_AUG = A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        # A.ElasticTransform(alpha=20, sigma=5, alpha_affine=10, p=0.5),
        A.ElasticTransform(alpha=20, sigma=5, p=0.5),
        A.Affine(translate_percent=0.05, rotate=(-5,5), scale=(0.95,1.05), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),
    ])

def simulate_attribute_edit(img_np, face_parts):
    """2_0_0: Attribute-Edit. 마스크를 활용해 특정 부위를 자연스럽게 변형."""
    if not face_parts: return img_np
    
    exist_parts = list(face_parts.keys())
    exist_parts = [part for part in exist_parts if part in ['face', 'eyes', 'brows', 'mouth', 'nose']]
    
    part_key = random.choice(exist_parts)
    mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    contour = face_parts[part_key]
    scale = random.uniform(0.8, 1.2)
    node_contour = random.randint(3,15)
    contour = scale_contour(contour, scale)
    # contour를 약간의 지터를 추가
    jitter = int(contour.shape[0] * 0.1)  # contour의 10% 크기
    contour = perturb_contour(contour, jitter=jitter, jitter_nodes=node_contour)
    cv2.fillPoly(mask, [contour], 255)
        
    # 마스크 경계 처리 정도 선택 약하게, 중간, 강하게
    if random.random() < 0.33:
        # 약하게: 경계 흐림
        mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0) / 255.0
    elif random.random() < 0.66:
        # 중간: 경계 흐림 + 투명도 조절
        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    else:
        # 강하게: 경계 흐림 + 투명도 조절 + 블러 강하게
        mask_blurred = cv2.GaussianBlur(mask, (21,21), 0) / 255.0
    # mask affine 변형
    if random.random() < 0.5:
        mask_blurred = cv2.warpAffine(mask_blurred, cv2.getRotationMatrix2D((img_np.shape[1]/2, img_np.shape[0]/2), random.uniform(-10, 10), random.uniform(0.9, 1.1)), (img_np.shape[1], img_np.shape[0]))
    del contour, scale, node_contour, jitter, mask
    # transformed = A.Compose([
    #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    #     A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    #     A.ElasticTransform(alpha=20, sigma=5, alpha_affine=10, p=0.5),
    #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),
    # ])(image=img_np)['image']
    transformed = ATTRIBUTE_EDIT_AUG(image=img_np)['image']
    mask_3ch = mask_blurred[..., np.newaxis]
    return (img_np * (1 - mask_3ch) + transformed * mask_3ch).astype(np.uint8)
def get_blend_mask(mask):
    H,W=mask.shape
    size_h=np.random.randint(192,257)
    size_w=np.random.randint(192,257)
    mask=cv2.resize(mask,(size_w,size_h))
    kernel_1=random.randrange(5,26,2)
    kernel_1=(kernel_1,kernel_1)
    kernel_2=random.randrange(5,26,2)
    kernel_2=(kernel_2,kernel_2)
    
    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
    del size_h, size_w, kernel_1, kernel_2, mask
    return mask_blured.reshape((mask_blured.shape+(1,)))

FACESWAP_AUG = A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.ElasticTransform(alpha=20, sigma=5, p=0.5),
        A.Affine(translate_percent=0.05, rotate=(-5,5), scale=(0.95,1.05), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),
    ])

def simulate_faceswap(src_img, face_parts):
    """2_0_1: Face-Swap. face_parts를 활용하여 얼굴 영역을 합성."""

    h, w = src_img.shape[:2]
    # 1) face_parts로 얼굴 영역 마스크 생성
    mask = np.zeros((h, w), dtype= np.uint8)
    contour = face_parts['face']
    scale = random.uniform(0.75, 1.05)
    node_contour = random.randint(3,15)
    contour = scale_contour(contour, scale)
    contour = perturb_contour(contour, jitter=int(contour.shape[0] * 0.1), jitter_nodes=node_contour)

    cv2.fillPoly(mask, [contour], 255)
    # 경계 부드럽게 페더링
    # blur 크기 랜덤하게 조정 및 타입 변경
    if random.random() < 0.5:
        blur_size = random.choice([(7, 7), (15, 15), (21, 21)])
        mask = cv2.GaussianBlur(mask, blur_size, 0) / 255.0
        
    else:
        mask = get_blend_mask(mask)        
        blend_list=[0.25,0.5,0.75,1,1,1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
        # mask 3채널로 확장 mask shape: (h, w, 1) -> (h, w, 3)
        if len(mask.shape) == 2:
            pass
        elif len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask * blend_ratio
    # mask affine 변형
    if random.random() < 0.5:
        mask = cv2.warpAffine(mask, cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-10, 10), random.uniform(0.9, 1.1)), (w, h))
    

    # 3) 얼굴 패치에 디지털 변형 (색·조명·기하 변형)
    face_patch = src_img.copy()
    # face_patch = A.Compose([
    #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    #     A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    #     A.ElasticTransform(alpha=20, sigma=5, alpha_affine=10, p=0.5),
    #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),
    # ])(image=face_patch)['image']
    face_patch = FACESWAP_AUG(image=face_patch)['image']
    if random.random() < 0.2:
        # horizontal flip
        face_patch = cv2.flip(face_patch, 1)
        mask = cv2.flip(mask, 1)
    # 4) 마스크를 이용한 부드러운 블렌딩
    # 최종 = (원본 * (1 - α)) + (변형된 얼굴 * α)
    blended = src_img * (1 - mask[..., None]) + face_patch * mask[..., None]
    
    # 잔여물 제거
    del face_patch, mask, contour, scale, node_contour
    return np.clip(blended, 0, 255).astype(np.uint8)


# 표정 변화(눈·입 움직임 등)가 과장되는 Deepfake 특성 흉내

# OpticalDistortion

# 프레임 경계에서 얼굴 윤곽이 살짝 왜곡되는 미세 굴절 효과 재현

# MotionBlur

# 비디오 특유의 움직임 블러로, 얼굴이 빠르게 움직일 때 나타나는 잔상 표현

# Soft Mask Blending

# 원본 ↔ 왜곡 이미지를 Gaussian-blur 마스크로 자연스럽게 합성

# 왜? Deepfake는 연속된 프레임에서 표정·움직임이 일관되지 않아 왜곡과 블러가 섞여 보이므로, Elastic+Optical+MotionBlur 조합이 효과적입니다.
MOTION_BLUR_KERNEL_1 = A.MotionBlur(blur_limit=(5, 15), p=1.0)
MOTION_BLUR_KERNEL_2 = A.MotionBlur(blur_limit=(5, 45), p=1.0)
def get_motion_mask(mask):
    H,W=mask.shape
    size_h=np.random.randint(192,257)
    size_w=np.random.randint(192,257)
    mask=cv2.resize(mask,(size_w,size_h))
    
    # mask_blured = cv2.MotionBlur(mask, kernel_1, 0)
    # mask_blured = A.MotionBlur(blur_limit=(5, 15), p=1.0)(image=mask)['image']
    mask_blured = MOTION_BLUR_KERNEL_1(image=mask)['image']
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured[mask_blured<1]=0
    
    # mask_blured = cv2.MotionBlur(mask_blured, kernel_2, np.random.randint(5,46))
    # mask_blured = A.MotionBlur(blur_limit=(5, 45), p=1.0)(image=mask_blured)['image']
    mask_blured = MOTION_BLUR_KERNEL_2(image=mask_blured)['image']
    mask_blured = mask_blured/(mask_blured.max())
    mask_blured = cv2.resize(mask_blured,(W,H))
    del size_h, size_w, mask
    return mask_blured.reshape((mask_blured.shape+(1,)))

VIDEO_DRIVEN_AUG = A.Compose([
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
        A.ElasticTransform(alpha=20, sigma=5,  p=0.5),
        A.Affine(translate_percent=0.05, rotate=(-5,5), scale=(0.95,1.05), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),
    ])
def simulate_video_driven(src_img, face_parts):
    """2_0_2: Video-Driven (Deepfake). face_parts를 활용해 표정을 과장되게 왜곡 및 모션 블러 추가"""
    h, w = src_img.shape[:2]
    # 1) face_parts로 얼굴 영역 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    contour = face_parts['face']
    scale = random.uniform(0.75, 1.05)
    node_contour = random.randint(3,15)
    contour = scale_contour(contour, scale)
    contour = perturb_contour(contour, jitter=int(contour.shape[0] * 0.1), jitter_nodes=node_contour)
    cv2.fillPoly(mask, [contour], 255)
    # 경계 부드럽게 페더링
    # blur 크기 랜덤하게 조정 및 타입 변경
    if random.random() < 0.5:
        blur_size = random.choice([(3, 3), (7, 7), (15, 15), (21, 21)])
        mask_1 = cv2.GaussianBlur(mask, blur_size, 0) / 255.0
        # Motion Blur 적용
        # mb = A.MotionBlur(blur_limit=(5, 15), p=1.0)
        mask_2 = MOTION_BLUR_KERNEL_1(image=mask)['image']
        mask = np.clip(mask_2, 0, 1)
        
    else:
        mask = get_motion_mask(mask)        
        blend_list=[0.25,0.5,0.75,1,1,1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
        # mask 3채널로 확장 mask shape: (h, w, 1) -> (h, w, 3)
        if len(mask.shape) == 2:
            pass
        elif len(mask.shape) == 3:
            mask = mask[:, :, 0]
        mask = mask * blend_ratio
    # mask affine 변형
    if random.random() < 0.5:
        mask = cv2.warpAffine(mask, cv2.getRotationMatrix2D((w/2, h/2), random.uniform(-10, 10), random.uniform(0.9, 1.1)), (w, h))
    

    # 3) 얼굴 패치에 디지털 변형 (색·조명·기하 변형)
    face_patch = src_img.copy()
    # face_patch = A.Compose([
    #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
    #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    #     A.GaussianBlur(blur_limit=(3, 5), p=0.5),
    #     A.ElasticTransform(alpha=20, sigma=5, alpha_affine=10, p=0.5),
    #     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.7),
    # ])(image=face_patch)['image']
    face_patch = VIDEO_DRIVEN_AUG(image=face_patch)['image']
    if random.random() < 0.2:
        # horizontal flip
        face_patch = cv2.flip(face_patch, 1)
        mask = cv2.flip(mask, 1)
    # 4) 마스크를 이용한 부드러운 블렌딩
    # 최종 = (원본 * (1 - α)) + (변형된 얼굴 * α)
    blended = src_img * (1 - mask[..., None]) + face_patch * mask[..., None]
    # 잔여물 제거
    del face_patch, mask, contour, scale, node_contour, src_img, face_patch
    # 5) 최종 이미지 반환
    return np.clip(blended, 0, 255).astype(np.uint8)
# Patch 노이즈 삽입
import numpy as np
import cv2
import random

def simulate_adv_random(img_np):
    """
    Adversarial simulation: per-pixel random perturbation without model.
    Adversarial perturbation on FFT representation.
    """
    
    img_FFT = np.fft.fft2(img_np.astype(np.float32) / 255.0)
    # Generate random noise in the frequency domain
    epsilon = random.uniform(1,8)  # Random epsilon for each image
    noise = np.random.uniform(-epsilon, epsilon, img_FFT.shape).astype(np.float32)
    # Apply noise to the FFT representation
    adv_FFT = img_FFT + noise
    # Inverse FFT to get the adversarial image
    adv_img = np.fft.ifft2(adv_FFT).real
    # Clip and convert back to uint8
    adv_img = np.clip(adv_img, 0, 1)
    del img_FFT, noise, adv_FFT
    return (adv_img * 255).astype(np.uint8)
def simulate_adv_edge(img_np):
    """
    Edge-aware adversarial simulation: perturbs along image gradients.
    Uses Sobel gradient as proxy for sensitive features.
    """
    img = img_np.astype(np.float32) / 255.0
    epsilon = random.uniform(2/255, 8/255)  # Random epsilon for each image
    # convert to grayscale for gradient
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # sign gradients
    grad_sign = np.sign(np.stack([gx, gy], axis=-1))
    # expand to channels
    grad_dir = np.concatenate([grad_sign[..., :1], grad_sign[..., 1:2], grad_sign[..., :1]], axis=2)
    # apply perturbation
    noise = epsilon * grad_dir
    adv = np.clip(img + noise, 0, 1)
    del gray, gx, gy, grad_sign, grad_dir, noise
    return (adv * 255).astype(np.uint8)


def simulate_adv_structured(img_np):
    """
    Structured adversarial simulation: block-wise perturbation.
    Divides image into blocks and assigns random sign per block.
    """
    img = img_np.astype(np.float32) / 255.0
    h, w, c = img.shape
    adv = img.copy()
    # Determine block size based on image dimensions
    epsilon = random.uniform(1/255, 8/255)  # Random epsilon for each block
    if img.shape[0] < 8 or img.shape[1] < 8:
        block_size = 8
    elif img.shape[0] < 16 or img.shape[1] < 16:
        block_size = 16
    elif img.shape[0] < 32 or img.shape[1] < 32:
        block_size = 32
    else:
        block_size = random.choice([8, 16, 32])
    # iterate blocks
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            sign = random.choice([-1, 1])
            y2 = min(h, y + block_size)
            x2 = min(w, x + block_size)
            adv[y:y2, x:x2, :] += sign * epsilon
    adv = np.clip(adv, 0, 1)
    del img, h, w, c, block_size, y, x, y2, x2, sign
    return (adv * 255).astype(np.uint8)

PIXEL_LEVEL_ADV_AUG = A.Compose([A.GaussianBlur(p=0.4, blur_limit=(3, 5)),
                           A.OneOf([
            A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="jpeg", p=1.0),
            A.ImageCompression(quality_lower=50, quality_upper=100, compression_type="webp", p=1.0),
        ], p=0.5),])
# Example unified interface
def simulate_pixel_level_adv(img_np):
    """
    Simulate pixel-level adversarial perturbation without model.
    mode: 'random', 'edge', or 'structured'.
    kwargs: parameters for each mode.
    """
    choose_mode = random.choice(['random', 'edge', 'structured'])
    if choose_mode == 'random':
        # print(f"Simulating random adversarial perturbation.")
        result_img= simulate_adv_random(img_np)
    if choose_mode == 'edge':
        # print(f"Simulating edge-aware adversarial perturbation.")
        result_img= simulate_adv_edge(img_np)
    if choose_mode == 'structured':
        # print(f"Simulating structured adversarial perturbation.")
        result_img= simulate_adv_structured(img_np)

    # --- 4. 최종 품질 저하 ---
    # aug_final = A.Compose([A.GaussianBlur(p=0.4, blur_limit=(3, 5)),
    #                        A.ImageCompression(quality_lower=50, quality_upper=80, p=0.5),])
    
    
    # result_img = aug_final(image=result_img)['image']
    result_img = PIXEL_LEVEL_ADV_AUG(image=result_img)['image']

    return result_img

# RandomShadow / Rain / Fog / Snow / SunFlare

# 자연현상(그림자·비·안개·눈·태양광) 중 하나를 랜덤 적용해 “환경 변화” 공격

# 얼굴 부위 Occlusion

# 눈 영역에 반투명 블록(선글라스 형태)으로 추가 손상

# 최종 GaussianBlur

# 전체 톤을 살짝 흐리게 해 삽입 이질감 완화

# 왜? 의미 단위 공격은 “진짜 같은 환경 변화”가 모델을 교란하기 때문에, 다양한 날씨·조명 변화와 부분 occlusion을 혼합합니다.
def simulate_semantic_level_adv(img_np, face_parts=None):
    """
    Alternative Semantic-Level Attack:
      1) 환경 변화 중 하나 선택:
         - directional_light: 얼굴에 비스듬한 빛 그라디언트
         - white_balance: 따뜻하거나 차가운 화이트 밸런스 이동
         - vignette: 화면 모서리 암화(비네팅)
      2) 얼굴 영역 Occlusion (확률 40%):
         - mesh: 반투명 격자망을 얼굴에 덮음
         - mask: 코·입 영역에 부드러운 수술용 마스크 형태 오클루전
      3) 눈 부위 Highlight Flare (확률 20%):
         - 좌·우 눈 중심에 작은 화이트 점광원 + 블러
      4) 최종 처리:
         - 가벼운 GaussianBlur
         - 약한 밝기/대비 변화(Color Jitter)
    """
    h, w = img_np.shape[:2]
    res = img_np.astype(np.float32)

    # 1) 환경 변화
    env = random.choice(['directional_light', 'white_balance', 'vignette'])
    if env == 'directional_light':
        # 비스듬한 밝기 그라디언트 생성
        angle = random.uniform(-math.pi/3, math.pi/3)
        xs = np.linspace(-1, 1, w)
        ys = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(xs, ys)
        proj = xv * math.cos(angle) + yv * math.sin(angle)
        norm = (proj - proj.min()) / (proj.max() - proj.min())
        overlay = (norm[..., None] * random.uniform(30, 80)).astype(np.float32)
        res = np.clip(res + overlay, 0, 255)
        del angle, xs, ys, xv, yv, proj, norm, overlay
    elif env == 'white_balance':
        # 따뜻하거나 차가운 색온도 이동
        shifts = random.choice([(-10, 0, 10), (10, 0, -10), (0, 5, -5)])
        for c, shift in enumerate(shifts):
            res[..., c] = np.clip(res[..., c] + shift, 0, 255)
        del shifts, c, shift
    else:  # vignette
        # 중앙은 그대로, 모서리로 갈수록 어둡게
        xs = np.linspace(-1, 1, w)
        ys = np.linspace(-1, 1, h)
        xv, yv = np.meshgrid(xs, ys)
        mask = 1 - (xv**2 + yv**2)
        mask = np.clip(mask, 0, 1)[..., None]
        res = res * mask + img_np.astype(np.float32) * (1 - mask)
        del xs, ys, xv, yv, mask
    # 2) 얼굴 영역 Occlusion
    if face_parts and 'face' in face_parts and random.random() < 0.4:
        choice = random.choice(['mesh', 'mask'])
        if choice == 'mesh':
            # 얼굴 convex hull 영역에 그리드 오버레이
            hull = cv2.convexHull(np.squeeze(face_parts['face']))
            face_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(face_mask, hull, 255)

            grid = np.zeros((h, w), np.uint8)
            spacing = random.randint(8, 16)
            for x in range(0, w, spacing):
                cv2.line(grid, (x, 0), (x, h), 255, 1)
            for y in range(0, h, spacing):
                cv2.line(grid, (0, y), (w, y), 255, 1)

            grid_mask = (grid > 0) & (face_mask > 0)
            # 그리드를 얼굴 영역에 반투명 회색으로 블렌딩
            res[grid_mask] = res[grid_mask] * 0.5 + 200
            del hull, face_mask, grid, spacing, x, y, grid_mask
        else:
            # 코·입 bounding box에 부드러운 마스크 오클루전
            if 'nose' in face_parts and 'mouth' in face_parts:
                npts = np.squeeze(face_parts['nose'])
                mpts = np.squeeze(face_parts['mouth'])
                x0, y0, w0, h0 = cv2.boundingRect(np.vstack((npts, mpts)))
                pad = 10
                x0, y0 = max(x0 - pad, 0), max(y0 - pad, 0)
                x1, y1 = min(x0 + w0 + 2*pad, w), min(y0 + h0 + 2*pad, h)

                rect = np.zeros((h, w), np.uint8)
                cv2.rectangle(rect, (x0, y0), (x1, y1), 255, -1)
                rect_blur = cv2.GaussianBlur(rect.astype(np.float32)/255, (51, 51), 15)[..., None]

                res = res * (1 - rect_blur * 0.7) + 230 * rect_blur * 0.7
                del npts, mpts, x0, y0, w0, h0, pad, x1, y1, rect, rect_blur
    # 3) 눈 부위 Highlight Flare
    if face_parts and 'l_eye' in face_parts and 'r_eye' in face_parts and random.random() < 0.2:
        for eye in ['l_eye', 'r_eye']:
            pts = np.squeeze(face_parts[eye])
            ex, ey, ew, eh = cv2.boundingRect(pts)
            cx, cy = ex + ew//2, ey + eh//2
            radius = random.randint(3, 6)
            cv2.circle(res, (cx, cy), radius, (255, 255, 255), -1)
            res = cv2.GaussianBlur(res, (5, 5), 0)
            del pts, ex, ey, ew, eh, cx, cy, radius
    elif face_parts and 'l_brow' in face_parts and 'r_brow' in face_parts and random.random() < 0.2:
        # 눈썹 영역에 작은 하이라이트 추가
        for brow in ['l_brow', 'r_brow']:
            pts = np.squeeze(face_parts[brow])
            ex, ey, ew, eh = cv2.boundingRect(pts)
            cx, cy = ex + ew//2, ey + eh//2
            # 투명한 원 그리기
            # brow 크기에 따라 반지름 조
            # brow 크기
            size_brow = max(ew, eh)
            radius = size_brow * random.uniform(0.4, 0.8)
            # 눈쪽으로 약간 이동
            cy += int(size_brow * random.uniform(0.3, 0.6))
            mask = np.zeros_like(res, dtype=np.uint8)
            # 눈썹 중심에 작은 원 그리기
            cv2.circle(mask, (cx, cy), int(radius), (255, 255, 255), -1)
            # 원을 블러 처리
            blur_size = random.sample([3, 5, 7], k=1)
            mask = cv2.GaussianBlur(mask, (blur_size[0], blur_size[0]), 0)
            # mask 안에 들어갈 distortion 
            mask_img = res.copy()
            mask_img = simulate_pixel_level_adv(mask_img)
            
            # 원 영역에 distortion 적용
            alpha = random.uniform(0.4, 0.8)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            mask = mask.astype(np.float32) / 255.0 * alpha
            res = res * (1 - mask) + mask_img * mask
            del pts, ex, ey, ew, eh, cx, cy, radius, mask, blur_size, mask_img, alpha
    # 4) 특정 영역에 Color Spotlight 적용
    if random.random() < 0.5:
        spot_num = random.sample([1, 1, 1, 2, 3], k=1)[0]  # 1~3개의 스포트라이트
        for _ in range(spot_num):
            # 랜덤한 위치에 원형 컬러 스포트라이트
            # 다양한 비정상 조명 색상 후보군 (BGR)
            light_colors = [
                (random.uniform(150, 200), random.uniform(50, 100), random.uniform(200, 255)), # 핑크/퍼플
                (random.uniform(200, 255), random.uniform(100, 150), random.uniform(50, 100)), # 시안/블루
                (random.uniform(50, 100), random.uniform(200, 255), random.uniform(150, 200)), # 그린/민트
                (random.uniform(220, 255), random.uniform(150, 200), random.uniform(50, 100)), # 오렌지/옐로우
            ]
            spot_color = random.choice(light_colors)
            # 랜덤 위치와 크기와 비율 (타원형)
            cx = random.randint(int(w*0.2), int(w*0.8))
            cy = random.randint(int(h*0.2), int(h*0.8))
            a = random.randint(int(w*0.1), int(w*0.3))
            b = random.randint(int(h*0.1), int(h*0.3))
            # 타원형 스포트라이트 생성
            spot_mask = np.zeros((h, w, 3), dtype=np.float32)
            cv2.ellipse(spot_mask, (cx, cy), (a, b), 0, 0, 360, spot_color, -1)
            # 주변 번짐 효과
            cv2.GaussianBlur(spot_mask, (15, 15), 0, spot_mask)
            # 원본 이미지에 스포트라이트 적용
            res = np.clip(res * 0.7 + spot_mask * 0.3, 0, 255)
            del spot_color, cx, cy, a, b, spot_mask
    if random.random() < 0.5:
        # --- 2. 랜덤 컬러 포인트 라이트 (레이저 닷) ---
        target_x, target_y = -1, -1
        if face_parts and face_parts.get('l_eye') is not None and face_parts.get('r_eye') is not None:
            l_eye_pts = np.squeeze(face_parts.get('l_eye'))
            r_eye_pts = np.squeeze(face_parts.get('r_eye'))
            all_eye_pts = np.vstack([l_eye_pts, r_eye_pts])
            ex, ey, ew, eh = cv2.boundingRect(all_eye_pts)
            target_x, target_y = random.randint(ex, ex + ew), random.randint(ey, ey + eh)
            del l_eye_pts, r_eye_pts, all_eye_pts, ex, ey, ew, eh
        else:
            target_x, target_y = random.randint(w//3, w*2//3), random.randint(h//3, h*2//3)
         # 포인트 라이트 색상도 랜덤화
        laser_color = random.choice([
            (220, 50, 255), (50, 255, 50), (255, 100, 100), (255, 255, 100)
        ])
        glow_color = tuple(c * 0.7 for c in laser_color) # 번짐은 살짝 어둡게
        center_color = (min(255, laser_color[0]+50), min(255, laser_color[1]+50), min(255, laser_color[2]+50)) # 중심은 더 밝게
        # 타겟 위치에 작은 원형 포인트 라이트 생성 (alpha 블렌딩)
        laser_mask = np.zeros((h, w, 3), dtype=np.float32)
        cv2.circle(laser_mask, (target_x, target_y), random.randint(3, 6), laser_color, -1)
        # 중심부는 더 밝게
        cv2.circle(laser_mask, (target_x, target_y), random.randint(1, 3), center_color, -1)
        # 주변은 약간 흐릿하게
        cv2.GaussianBlur(laser_mask, (9, 9), 0, laser_mask)
        # 주변 번짐 효과
        cv2.circle(laser_mask, (target_x, target_y), random.randint(6, 12), glow_color, -1)
        # 주변은 약간 흐릿하게
        cv2.GaussianBlur(laser_mask, (15, 15), 0, laser_mask)
        # 원본 이미지에 포인트 라이트 적용
        res = np.clip(res * 0.7 + laser_mask * 0.3, 0, 255)
        del target_x, target_y, laser_color, glow_color, center_color, laser_mask
    # 5) 최종 Blur + Color Jitter
    res = cv2.GaussianBlur(res, (3, 3), 0.5)
    alpha = random.uniform(0.9, 1.1)   # 대비
    beta  = random.uniform(-10, 10)    # 밝기
    res = np.clip(res * alpha + beta, 0, 255)
    return res.astype(np.uint8)

# BilateralFilter: 표면은 매끄럽게, 엣지 선명하게 유지 → GAN 표면 질감

# Upsample→Nearest: checkerboard 패턴 → 업샘플링 아티팩트

# Color Quantization + Posterize: 색상 계단 현상 → 부드럽지 못한 그라데이션

# JPEG Compression: 블록 노이즈 및 링잉 → 생성물 압축 흔적

# Subtle Noise: σ≈2 정도 미세 노이즈 → 과도하게 깔끔한 합성 방지

# Blend with Original: 원본 60–80% + 합성 20–40% → 완전 편집 아닌 “아이덴티티 유지” 연출

# 왜? GAN 생성물은 대체로 표면은 매끄럽지만, 색 밴딩과 압축 아티팩트를 동반하므로 이 단계별 증강이 실제와 유사합니다.
def simulate_id_consistent_gen(img_np):
    """2_2_0 Enhanced: ID-Consistent Generation.
    - Bilateral Filter → 표면 매끄럽게
    - Up/Down-sampling Artifacts → checkerboard 패턴
    - Color Banding → 색상 계단 현상
    - JPEG Compression → 실제 GAN 이미지 압축 흔적
    - Subtle Noise → 모델 과적합 방지용 약한 노이즈
    """
    h, w = img_np.shape[:2]

    # 1) 표면 매끄럽게: Bilateral Filter
    smooth = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)

    # 2) Checkerboard Artifact: 저해상도 → 고해상도
    scale = random.choice([2, 4])
    lr = cv2.resize(smooth, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
    checker = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3) Color Banding: 색상 단계 제한
    levels = random.randint(4, 8)
    quant = 256 // levels
    banded = ((checker // quant) * quant).astype(np.uint8)

    # 4) Posterize (추가 밴딩)
    banded = A.Posterize(num_bits=random.randint(4,6), p=1.0)(image=banded)['image']

    # 5) JPEG Compression Artifact
    compressed = A.ImageCompression(quality_lower=50, quality_upper=90, p=0.8)(image=banded)['image']

    # 6) Subtle Gaussian Noise
    noise = np.random.normal(0, 2, (h, w, 3)).astype(np.float32)
    noisy = np.clip(compressed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 7) 최종 Blend: 원본과 약간 섞어서 자연스러운 과도기 효과
    alpha = random.uniform(0.2, 0.4)
    out = cv2.addWeighted(img_np, 1 - alpha, noisy, alpha, 0)

    return out
# Blurred style source: 랜덤 노이즈보다 부드러운 패턴으로 스타일흉내

# FFT Mixing: 저주파만 스타일 영상에 할당 → 디테일은 원본 유지

# Bilateral + MedianBlur: 역변환 잡음 제거

# Low-weight Blend (0.1): 미묘한 스타일 전이, 너무 강하지 않게

# 왜? 저주파 스타일 전이 방식은 콘텐츠 보존과 스타일 주입 균형이 중요해, FFT+블러 조합이 간단하면서 효과적입니다.
def simulate_style_transfer_gen(img_np):
    """2_2_1 Enhanced: Style Transfer Generation with reduced noise and smoother transitions."""
    h, w, _ = img_np.shape

    # 1) Create style source as a blurred version of the original (reduces noise)
    style_src = cv2.GaussianBlur(img_np, (51, 51), 0)

    # 2) FFT of grayscale images
    src_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    style_gray = cv2.cvtColor(style_src, cv2.COLOR_RGB2GRAY)
    src_fft = np.fft.fftshift(np.fft.fft2(src_gray))
    style_fft = np.fft.fftshift(np.fft.fft2(style_gray))

    # 3) Low-frequency mask (random small radius)
    r = random.randint(5, 15)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), r, 1, -1)
    mask_3ch = mask[..., None]

    # 4) Mix FFTs: low-frequency from style, high-frequency from content
    mixed_fft = src_fft * (1 - mask) + style_fft * mask

    # 5) Inverse FFT to gray
    inv = np.fft.ifft2(np.fft.ifftshift(mixed_fft))
    new_gray = np.abs(inv)
    new_gray = np.clip(new_gray, 0, 255).astype(np.uint8)

    # 6) Smooth the result to reduce noise
    new_gray = cv2.bilateralFilter(new_gray, d=9, sigmaColor=75, sigmaSpace=75)
    new_gray = cv2.medianBlur(new_gray, 5)

    # 7) Blend back into color image with low style weight
    styled_rgb = cv2.cvtColor(new_gray, cv2.COLOR_GRAY2RGB)
    out = cv2.addWeighted(img_np, 0.7, styled_rgb, 0.3, 0)

    return out
# Two-tone Gradient: 텍스트 프롬프트에서 자주 등장하는 “컬러 라이트” 효과

# Radial Vignette: 모호한 경계, 프레임 중심 집중

# Colored Rim Light: “테두리 조명” (“teal rim light” 등) 연출

# 가중치 조정: 전체적 톤을 0–255로 안정적으로 마무리

# 왜? 텍스트 유도 생성 모델에서 자주 쓰이는 “dramatic lighting” 묘사를, 컬러 그라데이션+비네팅+림 라이트로 시각화했습니다.
def simulate_prompt_driven_gen(img_np):
    """2_2_2 Enhanced: Prompt‐Driven Generation with unreal lighting effects."""
    h, w, _ = img_np.shape

    # 1) Two colored light gradient (left→right)
    light1_color = np.array([random.uniform(0.6, 1.0) for _ in range(3)], dtype=np.float32)
    light2_color = np.array([random.uniform(0.6, 1.0) for _ in range(3)], dtype=np.float32)
    mask_lr = np.linspace(1, 0, w, dtype=np.float32)[None, :, None]  # H×W×1
    lit = img_np.astype(np.float32)
    lit = lit * (mask_lr * light1_color + (1 - mask_lr) * light2_color)

    # 2) Radial vignette (darken corners)
    xv, yv = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
    radius = np.sqrt(xv**2 + yv**2)
    vignette = np.clip(1 - radius, 0.3, 1.0)[..., None]
    lit *= vignette

    # 3) Colored rim light (subtle halo)
    rim = np.exp(-((radius - 0.8)*10)**2)  # ring around edges
    rim_color = np.array([random.uniform(0.8,1.0), random.uniform(0.6,0.8), random.uniform(0.6,0.8)])
    lit += (rim[...,None] * rim_color * 80)  # boost by up to ~80 intensity

    # 4) Final tone‐mapping & clip
    out = np.clip(lit, 0, 255).astype(np.uint8)
    return out

import cv2
import numpy as np
import random

def simulate_mask_trans(img_np, face_parts):
    """
    [Final Ver. 2] 얼굴 가시성을 확보하고, 누락된 랜드마크에 대한 안정성을 강화한 최종 버전입니다.
    - 전체 효과의 투명도를 조절하여 얼굴이 잘 보이도록 합니다.
    - 'nose', 'brows' 데이터가 없어도 'face' 데이터만으로 동작하도록 수정합니다.
    """
    h, w = img_np.shape[:2]

    # --- 파라미터 조절 (얼굴이 잘 보이도록 전체적으로 강도를 낮춤) ---
    # 1. 마스크의 전체적인 최종 투명도 (가장 중요한 제어 변수)
    MASK_OPACITY = random.uniform(0.2, 0.5)  # 이 값을 낮추면 더 투명해집니다.

    # 2. 개별 효과의 강도 (과하지 않게 조정)
    HIGHLIGHT_INTENSITY = random.uniform(0.5, 0.8)
    SHADOW_INTENSITY = random.uniform(0.4, 0.7)
    DISTORTION_STRENGTH = random.uniform(1.2, 2.0)
    FOG_OPACITY = random.uniform(0.05, 0.15)

    # --- 1. 안정적인 랜드마크 추출 ---
    # .get()을 사용하여 키가 없어도 오류가 발생하지 않도록 함
    face_contour_data = face_parts.get('face')
    if face_contour_data is None:
        print("오류: 'face' 키가 없어 마스크를 생성할 수 없습니다.")
        return img_np # 필수 데이터가 없으면 원본 반환
    
    face_contour = np.squeeze(face_contour_data).astype(np.int32)

    # 코와 눈썹 데이터는 선택적으로 사용
    nose_pts_data = face_parts.get('nose')
    nose_pts = np.squeeze(nose_pts_data).astype(np.int32) if nose_pts_data is not None else None
    
    brows_pts_data = face_parts.get('brows')
    brows_pts = np.squeeze(brows_pts_data).astype(np.int32) if brows_pts_data is not None else None

    # --- 2. 마스크 모양 정의 (데이터 유무에 따라) ---
    alpha_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(alpha_mask, [face_contour], 1.0)
    # 코 데이터가 있을 경우에만 코 영역을 추가로 채움
    if nose_pts is not None:
        cv2.fillConvexPoly(alpha_mask, nose_pts, 1.0)
    
    alpha_mask_blurred = cv2.GaussianBlur(alpha_mask, (61, 61), 0)

    # --- 3. 동적 하이라이트 생성 (데이터 없으면 대체 위치 사용) ---
    highlight_map = np.zeros((h, w, 3), dtype=np.float32)
    fx, fy, fw, fh = cv2.boundingRect(face_contour)
    highlight_points = {}

    # 이마: 눈썹 데이터가 있으면 사용, 없으면 얼굴 윤곽선 기준으로 대체 위치 계산
    if brows_pts is not None:
        forehead_center = (int(brows_pts[:, 0].mean()), int(brows_pts[:, 1].min() - fh * 0.05))
    else:
        forehead_center = (int(fx + fw * 0.5), int(fy + fh * 0.15))
    highlight_points['forehead'] = forehead_center

    # 코: 코 데이터가 있으면 사용, 없으면 얼굴 중앙을 대체 위치로 사용
    if nose_pts is not None:
        nose_center = tuple(np.mean(nose_pts, axis=0).astype(int))
    else:
        nose_center = (int(fx + fw * 0.5), int(fy + fh * 0.5))
    highlight_points['nose_bridge'] = nose_center

    # 뺨과 턱은 'face' 윤곽선만으로 계산 가능하므로 안정적
    highlight_points['left_cheek'] = (int(fx + fw * 0.25), int(fy + fh * 0.45))
    highlight_points['right_cheek'] = (int(fx + fw * 0.75), int(fy + fh * 0.45))
    highlight_points['chin'] = tuple(face_contour[np.argmax(face_contour[:, 1])])
    
    for _, center in highlight_points.items():
        size_factor = w * random.uniform(0.05, 0.12)
        axis = (int(size_factor), int(size_factor * random.uniform(0.25, 0.5)))
        cv2.ellipse(highlight_map, center, axis, random.randint(0, 180), 0, 360, (1, 1, 1), -1)

    highlight_map = cv2.GaussianBlur(highlight_map, (31, 31), 0) # 블러를 살짝 더 줌
    highlight_map *= (alpha_mask_blurred[..., None] * HIGHLIGHT_INTENSITY)

    # --- 4. 그림자, 왜곡, 안개 효과 생성 (로직은 이전과 유사) ---
    edge_map = cv2.Laplacian(alpha_mask_blurred, cv2.CV_32F)
    edge_map = np.clip(edge_map, 0, 1)
    shadow_map = cv2.GaussianBlur(edge_map, (31, 31), 0)
    shadow_map = 1.0 - (shadow_map / (np.max(shadow_map) + 1e-6) * SHADOW_INTENSITY)
    shadow_map = shadow_map[..., None]

    height_map = cv2.GaussianBlur(alpha_mask, (121, 121), 0)
    sobel_x, sobel_y = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=31), cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=31)
    dx, dy = sobel_x * DISTORTION_STRENGTH, sobel_y * DISTORTION_STRENGTH
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    distorted_img = cv2.remap(img_np, (map_x - dx).astype(np.float32), (map_y - dy).astype(np.float32), cv2.INTER_LINEAR)

    fog_map = cv2.resize(np.random.normal(0.5, 0.5, (h // 40, w // 40)), (w, h))
    fog_map = cv2.GaussianBlur(fog_map, (151, 151), 0)
    fog_map = (fog_map - np.min(fog_map)) / (np.max(fog_map) - np.min(fog_map) + 1e-6)
    fog_map = (fog_map * alpha_mask_blurred * FOG_OPACITY)[..., None]
    
    # --- 5. 최종 합성 (가시성을 고려한 새로운 방식) ---
    # 1. 마스크 효과가 적용될 레이어 생성 (왜곡, 그림자, 하이라이트, 안개 포함)
    mask_layer_float = (distorted_img.astype(np.float32) / 255.0) * shadow_map
    mask_layer_float += highlight_map
    mask_layer_float += fog_map
    mask_layer_float = np.clip(mask_layer_float, 0, 1)

    # 2. 원본 이미지와 마스크 레이어를 최종 투명도(MASK_OPACITY)로 합성
    img_float = img_np.astype(np.float32) / 255.0
    final_alpha = alpha_mask_blurred[..., None] * MASK_OPACITY
    
    # Affine Transform 을 통해서, random하게 왜곡된 마스크 레이어를 적용
    mask_layer_float = cv2.warpAffine(
        mask_layer_float,
        cv2.getRotationMatrix2D((w // 2, h // 2), random.uniform(-10, 10), random.uniform(0.9, 1.1)),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
        
    # 원본 이미지 * (1 - 투명도) + 마스크 레이어 * (투명도)
    final_image_float = img_float * (1 - final_alpha) + mask_layer_float * final_alpha
    
    final_image = (final_image_float * 255).astype(np.uint8)
    
    return final_image