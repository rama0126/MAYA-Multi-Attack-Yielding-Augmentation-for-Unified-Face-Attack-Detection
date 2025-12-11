import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from utils import parse_protocol_file, get_live_samples  # parse_protocol_file expects fas_root_dir
import dlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
import random
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")


class FASDataset(Dataset):
    PHYSICAL_CODES = ['1_0_0','1_0_1','1_0_2','1_1_0','1_1_1','1_1_2']
    DIGITAL_CODES  = ['2_0_0','2_0_1','2_0_2','2_1_0','2_1_1','2_2_0','2_2_1','2_2_2']

    def __init__(self, samples_list=None, protocol_file=None, fas_root_dir=None, 
                 transform=None, pseudo_rate=0.3, img_size=256, 
                 live_aug_rate=0.5, spoof_aug_rate=0.5, use_ram=False, is_train=False):

        if samples_list:
            parsed = [(p, code, label) for p, code, label in samples_list]
        else:
            if fas_root_dir is None and protocol_file is not None : # Added for safety
                 # If fas_root_dir is not given, assume protocol_file contains absolute paths or paths relative to CWD
                 fas_root_dir = '.' # Or some other sensible default, or raise error
                 print(f"Warning: fas_root_dir not provided, using current directory '{fas_root_dir}' as root for protocol paths.")

            parsed = parse_protocol_file(protocol_file, fas_root_dir)
            
        # Duplicate real samples to increase their representation if needed, or for specific aug strategies
        # The original code did:
        # real_samples = [p for p, code, label in parsed if label == 0]
        # parsed += real_samples
        # This doubles the live samples. Let's keep it for now.
        if parsed and is_train:
            real_samples = [(p, code, label) for p, code, label in parsed if label == 0]
            parsed.extend(real_samples * 20) # Use extend for list of tuples
        self.samples = parsed
        if not self.samples:
            print("Warning: FASDataset initialized with 0 samples.")


        self.pseudo_rate = pseudo_rate
        self.live_aug_rate = live_aug_rate
        self.spoof_aug_rate = spoof_aug_rate
        
        # Base transform is applied last to the PIL image
        self.base_transform_tensor = transform
        
        self.pil_augmentations = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)) # PIL GaussianBlur sigma needs a range
        ])

        self.SBI_transform_pil = transforms.Compose([ # Strong Brightness/Inner transform for PIL
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.RandomAutocontrast(p=0.3),
        ])
        self.imgs = {}
        self.landmarks = {}
        self.use_ram = use_ram
        self.is_train = is_train
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if not self.samples: # Handle empty dataset
            raise IndexError("Dataset is empty")
            
        path, attack_code, label = self.samples[idx % len(self.samples)]

        try:
            if self.use_ram:
                if path in self.imgs:
                    img = self.imgs[path]
                else:
                    img = Image.open(path).convert('RGB')
                    self.imgs[path] = img
            else:
                img = Image.open(path).convert('RGB')
        except FileNotFoundError:
            print(f"Missing {path}, loading next sample.")
            return self.__getitem__((idx + 1)) # No modulo here, rely on __len__ for range
        except Exception as e:
            print(f"Error loading {path}: {e}, loading next sample.")
            return self.__getitem__((idx + 1))

        # Apply initial PIL augmentations that don't conflict with pseudo-attacks
        

        is_pseudo = False
        if self.is_train:
            if label == 0 and random.random() < self.pseudo_rate:
                if random.random() < 0.8:
                    subtype = random.choice(self.PHYSICAL_CODES)
                else:
                    subtype = random.choice(self.DIGITAL_CODES)
                img, success = self._apply_subtype_transform(img, subtype, path) # path is for landmarks
                if success:
                    label = 1 # It's now a pseudo-spoof
                    is_pseudo = True
                if not success:
                    label = 0
        
        # if not is_pseudo: # If it's an original live or original spoof, or pseudo creation failed
        #     if label == 0: # Original Live
        #         if random.random() < self.live_aug_rate:
        #             img = self.pil_augmentations(img) # Apply general augs to live
        #     else: # Original Spoof
        #         if random.random() < self.spoof_aug_rate:
        #             img = self.pil_augmentations(img) # Apply general augs to spoofs
        
        img = self.base_transform_tensor(img) # ToTensor and Normalize
        return img, torch.tensor(label, dtype=torch.long), is_pseudo # Use long for CrossEntropyLoss usually

    def _get_landmarks(self, img_pil, image_path_for_cache=""):
        """Helper to get landmarks, using a cache if image_path_for_cache is provided.
        Returns: list of (x, y) tuples as Python ints (not numpy types).
        """
        # Construct cache path if image_path_for_cache is meaningful
        landmark_cache_path = ""
        if image_path_for_cache:
            base, ext = os.path.splitext(image_path_for_cache)
            cache_dir = os.path.join(os.path.dirname(base), "landmark_cache")
            os.makedirs(cache_dir, exist_ok=True)
            landmark_cache_path = os.path.join(cache_dir, os.path.basename(base) + "_landmarks.npy")

        if image_path_for_cache in self.landmarks:
            landmarks = self.landmarks[image_path_for_cache]
            return landmarks
        
        
        if landmark_cache_path and os.path.exists(landmark_cache_path):
            try:
                landmarks = np.load(landmark_cache_path)
                # Ensure output is a list of tuples of ints (not numpy types)
                landmarks = [tuple(map(int, pt)) for pt in landmarks]
                return landmarks
            except Exception as e:
                print(f"Error loading cached landmarks {landmark_cache_path}: {e}")
        img_np = np.array(img_pil)
        # Ensure image is 8-bit gray or RGB for dlib
        if img_np.ndim == 2:
            pass
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            pass
        else:
            print("Unsupported image format for landmark detection.")
            return None

        faces = detector(img_np, 1)
        if len(faces) == 0:
            return None

        shape = predictor(img_np, faces[0])
        # Ensure output is a list of tuples of ints (not numpy types)
        landmarks = [(int(p.x), int(p.y)) for p in shape.parts()]

        # Always return a list of tuples of Python ints, not numpy ints
        landmarks = [tuple(int(x) for x in pt) for pt in landmarks]

        if landmark_cache_path:
            try:
                # Save as np.int32 for compatibility, but always return Python ints
                np.save(landmark_cache_path, np.array(landmarks, dtype=np.int32))
            except Exception as e:
                print(f"Error saving landmarks to {landmark_cache_path}: {e}")
        self.landmarks[image_path_for_cache] = landmarks
        return landmarks

    # --- Start of Pseudo-Attack Methods ---
    def _simulate_print(self, img, path):
        print_img = img.copy()
        ori_img = img.copy()
        flag = True
        if random.random() < 0.2:
            # JPEG Compression image
            print_img_np = np.array(print_img)
            quality = random.randint(50, 100)
            encoded_img = cv2.imencode('.jpg', print_img_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
            decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            print_img = Image.fromarray(decoded_img)
            flag = False
        elif random.random() < 0.2:
            # WebP Compression image
            print_img_np = np.array(print_img)
            quality = random.randint(50, 100)
            encoded_img = cv2.imencode('.webp', print_img_np, [int(cv2.IMWRITE_WEBP_QUALITY), quality])[1]
            decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            print_img = Image.fromarray(decoded_img)
            flag = False
        if random.random() < 0.8:
            # Brightness 조정
            enhancer = ImageEnhance.Brightness(print_img)
            brightness_factor = random.uniform(0.6, 1.4)  # [1-0.4, 1+0.4]
            print_img = enhancer.enhance(brightness_factor)
            
            # Contrast 조정
            enhancer = ImageEnhance.Contrast(print_img)
            contrast_factor = random.uniform(0.6, 1.4)
            print_img = enhancer.enhance(contrast_factor)
            
            # Saturation 조정
            enhancer = ImageEnhance.Color(print_img)
            saturation_factor = random.uniform(0.6, 1.4)
            print_img = enhancer.enhance(saturation_factor)
            flag = False
        if random.random() < 0.6:
            noise = np.random.normal(0, random.uniform(5, 15), print_img.size + (3,))
            noise_img = Image.fromarray(np.clip(np.array(print_img) + noise, 0, 255).astype(np.uint8))
            print_img = Image.blend(print_img, noise_img, alpha=random.uniform(0.1, 0.3))
            flag = False
        if random.random() < 0.4:
            filter_size = random.randint(3, 5)
            if filter_size % 2 == 0:
                filter_size += 1
            print_img = print_img.filter(ImageFilter.ModeFilter(size=filter_size))
            flag = False
        if random.random() < 0.05:
            # 모폴로지 연산
            kernel = np.ones((3,3), np.uint8)
            if random.random() < 0.5:
                print_img = cv2.erode(np.array(print_img), kernel, iterations=1)
            else:
                print_img = cv2.dilate(np.array(print_img), kernel, iterations=1)
            print_img = Image.fromarray(print_img)
            flag = False
        if random.random() < 0.2:
            # 블러 처리
            blur_radius = random.uniform(0.5, 1.5)
            print_img = print_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            flag = False
        if flag:
            return img, False
        return print_img, True

    def _simulate_replay(self, img, path):
        attack_img = img.copy()
        flag = True
        # 1. Add moiré pattern (more prominent for replay)
        if random.random() < 0.8: # Higher chance for replay
            attack_img = self._add_moire_pattern(attack_img, intensity=random.uniform(0.05, 0.15), step=random.randint(3,5))
            flag =False
        # 2. 디스플레이 화면의 백라이트 효과 시뮬레이션
        if random.random() < 0.5:
            # 중앙이 더 밝은 비네팅 효과
            width, height = attack_img.size
            vignette = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(vignette)
            
            for i in range(min(width, height) // 4):
                alpha = 255 - (i * 2)
                draw.ellipse([
                    width//2 - width//2 + i, height//2 - height//2 + i,
                    width//2 + width//2 - i, height//2 + height//2 - i
                ], fill=(alpha, alpha, alpha))
            
            vignette = vignette.filter(ImageFilter.GaussianBlur(radius=width//8))
            attack_img = Image.blend(attack_img, vignette, alpha=random.uniform(0.05, 0.15))
            flag = False
        # 3. 화면 반사 효과
        if random.random() < 0.4:
            reflection = np.random.randint(180, 255, size=attack_img.size + (3,), dtype=np.uint8)
            reflection_img = Image.fromarray(reflection, 'RGB')
            reflection_img = reflection_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(10, 20)))
            attack_img = Image.blend(attack_img, reflection_img, alpha=random.uniform(0.03, 0.08))
            flag = False
        if flag:
            return img, False
        return attack_img, True

    def _simulate_cutouts(self, img, path):
        img_orig = img.copy()
        img = img.copy()
        landmarks = self._get_landmarks(img, path)
        def _simulate_print(img, path):
                print_img = img.copy()
                ori_img = img.copy()
                flag = True
                if random.random() < 0.2:
                    # JPEG Compression image
                    print_img_np = np.array(print_img)
                    quality = random.randint(50, 100)
                    encoded_img = cv2.imencode('.jpg', print_img_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
                    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                    print_img = Image.fromarray(decoded_img)
                    flag = False
                elif random.random() < 0.2:
                    # WebP Compression image
                    print_img_np = np.array(print_img)
                    quality = random.randint(50, 100)
                    encoded_img = cv2.imencode('.webp', print_img_np, [int(cv2.IMWRITE_WEBP_QUALITY), quality])[1]
                    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
                    print_img = Image.fromarray(decoded_img)
                    flag = False
                if random.random() < 0.8:
                    # Brightness 조정
                    enhancer = ImageEnhance.Brightness(print_img)
                    brightness_factor = random.uniform(0.6, 1.4)  # [1-0.4, 1+0.4]
                    print_img = enhancer.enhance(brightness_factor)
                    
                    # Contrast 조정
                    enhancer = ImageEnhance.Contrast(print_img)
                    contrast_factor = random.uniform(0.6, 1.4)
                    print_img = enhancer.enhance(contrast_factor)
                    
                    # Saturation 조정
                    enhancer = ImageEnhance.Color(print_img)
                    saturation_factor = random.uniform(0.6, 1.4)
                    print_img = enhancer.enhance(saturation_factor)
                    flag = False
                if random.random() < 0.6:
                    noise = np.random.normal(0, random.uniform(5, 15), print_img.size + (3,))
                    noise_img = Image.fromarray(np.clip(np.array(print_img) + noise, 0, 255).astype(np.uint8))
                    print_img = Image.blend(print_img, noise_img, alpha=random.uniform(0.1, 0.3))
                    flag = False
                if random.random() < 0.4:
                    filter_size = random.randint(3, 5)
                    if filter_size % 2 == 0:
                        filter_size += 1
                    print_img = print_img.filter(ImageFilter.ModeFilter(size=filter_size))
                    flag = False
                if random.random() < 0.05:
                    # 모폴로지 연산
                    kernel = np.ones((3,3), np.uint8)
                    if random.random() < 0.5:
                        print_img = cv2.erode(np.array(print_img), kernel, iterations=1)
                    else:
                        print_img = cv2.dilate(np.array(print_img), kernel, iterations=1)
                    print_img = Image.fromarray(print_img)
                    flag = False
                if random.random() < 0.2:
                    # 블러 처리
                    blur_radius = random.uniform(0.5, 1.5)
                    print_img = print_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                    flag = False
                if flag:
                    return img, False
                return print_img, True
        if landmarks is None: return img_orig, False
        # "Photo" part undergoes print simulation
        if random.random() > 0.5:
            # Define eye and mouth regions from landmarks
            # Left eye: 36-41, Right eye: 42-47, Mouth: 48-67
            key_regions_indices = list(range(36, 48)) + list(range(48, 68))
            
            # Create a mask for the "photo" part (face excluding key regions)
            photo_mask = Image.new('L', img.size, 255)
            draw = ImageDraw.Draw(photo_mask)
            
            # Carve out eye/mouth regions from this mask
            for i_group in [list(range(36,42)), list(range(42,48)), list(range(48,68))]:
                region_pts = [landmarks[i] for i in i_group]
                if len(region_pts) > 2: # Need at least 3 points for a polygon
                    # Expand region slightly to ensure cut
                    np_pts = np.array(region_pts)
                    center = np.mean(np_pts, axis=0)
                    scaled_pts = center + 1.5 * (np_pts - center)
                    # Ensure coordinates are a list of tuples of Python ints
                    scaled_pts_int = [tuple(int(round(x)) for x in pt) for pt in scaled_pts.tolist()]
                    draw.polygon(scaled_pts_int, fill=0) # Cut out region
            
            
                photo_part, flag = _simulate_print(img_orig.copy(), path)
                if not flag:
                    photo_part, flag = _simulate_print(img_orig.copy(), path)
        
            # img_orig 에 photomask 부분에 photo_part 를 넣어줘야 함.
            img_result = img_orig.copy()
            img_result.paste(photo_part, mask=photo_mask)
        else:
            # Define eye and mouth regions from landmarks
            # Left eye: 36-41, Right eye: 42-47, Mouth: 48-67
            key_regions_indices = list(range(36, 48)) + list(range(48, 68))
            
            # Create a mask for the "photo" part (face excluding key regions)
            photo_mask = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(photo_mask)
            
            types = [
                [list(range(36,42)), list(range(42,48)), list(range(48,68))],
                [list(range(36,42)), list(range(42,48))],
                [list(range(36,42)), list(range(48,68))],
                [list(range(42,48)), list(range(48,68))],
                [list(range(36,42)), ],
                [list(range(42,48)), ],
                [list(range(48,68))],]
            types_s = random.choice(types)
            # Carve out eye/mouth regions from this mask
            for i_group in types_s:
                region_pts = [landmarks[i] for i in i_group]
                if len(region_pts) > 2: # Need at least 3 points for a polygon
                    # Expand region slightly to ensure cut
                    np_pts = np.array(region_pts)
                    center = np.mean(np_pts, axis=0)
                    scaled_pts = center + 1.1 * (np_pts - center)
                    # Ensure coordinates are a list of tuples of Python ints
                    scaled_pts_int = [tuple(int(round(x)) for x in pt) for pt in scaled_pts.tolist()]
                    draw.polygon(scaled_pts_int, fill=255) # Cut out region
            
            # img_orig 에 photomask 부분에 photo_part 를 넣어줘야 함.
            img_result = img_orig.copy()
            photo_part = img_orig.copy()
            # photo_part를 그림자 처리
            enhancer = ImageEnhance.Brightness(photo_part)
            brightness = random.uniform(0.5, 0.8)
            photo_part = enhancer.enhance(brightness) # 밝기 조절
            img_result.paste(photo_part, mask=photo_mask)
            
        return img_result, True

    def _simulate_transparent_mask(self, img, path):
        return img, False

    def _simulate_plaster_mask(self, img, path):
        img = img.copy()
        return img, False

    def _simulate_resin_mask(self, img, path):
        img = img.copy()
        flag = True
        # 1. Unnatural skin tone (color shift)
        r, g, b = img.split()
        r_new = r.point(lambda i: i * random.uniform(0.7, 1.3))
        g_new = g.point(lambda i: i * random.uniform(0.7, 1.3))
        b_new = b.point(lambda i: i * random.uniform(0.7, 1.3))
        img = Image.merge("RGB", (r_new, g_new, b_new))
        img = img.filter(ImageFilter.SMOOTH) # Blend color changes
        flag = False
        # 2. Glossy look: increase contrast, add subtle highlights
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(1.1, 1.5))
        # 3. Reduce fine skin texture but keep edges sharper than plaster
        img = img.filter(ImageFilter.MedianFilter(size=3))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        return img, True

    def _simulate_attribute_edit(self, img, path):
        # Use _landmark_perturbation or _texture_disturbance locally
        img, success = self._landmark_perturbation(img, path, perturb_specific_group=True)
        return img, success

    def _simulate_face_swap(self, img, path):
        # _self_blended_image is a good fit
        img, success = self._self_blended_image(img, path, stronger_inner_aug=True)
        return img, success

    def _simulate_video_driven(self, img, path):
        # _landmark_perturbation focusing on expressive regions
        img, success = self._landmark_perturbation(img, path)
        return img, success

    def _simulate_pixel_level(self, img, path): # Deepfake-like
        # Combine _texture_disturbance and perhaps some smoothing
        return img, False

    def _simulate_semantic_level(self, img, path):
        img, success = self._landmark_perturbation(img, path, perturb_expressive_only=True)
        return img, success

    def _simulate_id_consistent(self, img, path): # Advanced Deepfake
        # Similar to pixel_level but perhaps more subtle or focused on consistency artifacts
        img, success = self._self_blended_image(img, path, stronger_inner_aug=False, subtle_blend=True)
        return img, success

    def _simulate_style_transfer(self, img, path):
        return img, False
    def _simulate_prompt_driven(self, img, path):
        img_orig = img.copy()
        return img, False
    
    # --- End of Pseudo-Attack Methods ---

    def _apply_subtype_transform(self, img: Image.Image, subtype: str, path: str):
        """Calls the corresponding simulation method for the subtype."""
        original_img = img.copy() # Keep a copy in case simulation fails
        success = False

        if subtype == '1_0_0':   img, success = self._simulate_print(img, path)
        elif subtype == '1_0_1': img, success = self._simulate_replay(img, path)
        elif subtype == '1_0_2': img, success = self._simulate_cutouts(img, path)
        elif subtype == '1_1_0': img, success = self._simulate_transparent_mask(img, path)
        elif subtype == '1_1_1': img, success = self._simulate_plaster_mask(img, path)
        elif subtype == '1_1_2': img, success = self._simulate_resin_mask(img, path)
        elif subtype == '2_0_0': img, success = self._simulate_attribute_edit(img, path)
        elif subtype == '2_0_1': img, success = self._simulate_face_swap(img, path)
        elif subtype == '2_0_2': img, success = self._simulate_video_driven(img, path)
        elif subtype == '2_1_0': img, success = self._simulate_pixel_level(img, path)
        elif subtype == '2_1_1': img, success = self._simulate_semantic_level(img, path)
        elif subtype == '2_2_0': img, success = self._simulate_id_consistent(img, path)
        elif subtype == '2_2_1': img, success = self._simulate_style_transfer(img, path)
        elif subtype == '2_2_2': img, success = self._simulate_prompt_driven(img, path)
        else:
            # Unknown subtype, return original image and False
            return original_img, False
        
        return (img, success) if success else (original_img, False)


    # --- Existing Helper Methods (Adapted/Reviewed) ---
    def _add_moire_pattern(self, img, intensity=0.1, step=5): # Added intensity and step params
        arr = np.array(img).astype(np.float32)
        # Create two sine wave patterns
        x = np.arange(arr.shape[1])
        y = np.arange(arr.shape[0])
        X, Y = np.meshgrid(x, y)
        
        freq1 = random.uniform(0.1, 0.5)
        angle1 = random.uniform(0, np.pi)
        moire1 = np.sin(freq1 * (X * np.cos(angle1) + Y * np.sin(angle1)))
        
        freq2 = random.uniform(0.1, 0.5)
        angle2 = random.uniform(0, np.pi)
        if abs(angle1 - angle2) < 0.2 : angle2 = (angle1 + np.pi/2) % np.pi # Ensure different angles
        moire2 = np.sin(freq2 * (X * np.cos(angle2) + Y * np.sin(angle2)))
        
        # Combine and scale
        combined_moire = (moire1 + moire2) / 2.0 # Range -1 to 1
        
        # Apply to image channels. Intensity controls strength.
        # Max change of intensity * 127.5
        noise_val = intensity * 127.5
        for c in range(3):
            arr[:,:,c] += combined_moire * random.uniform(0.5 * noise_val, noise_val) # Modulate per channel

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _landmark_perturbation(self, img, path, perturb_specific_group=False, perturb_expressive_only=False):
        img_orig = img.copy()
        landmarks = self._get_landmarks(img_orig, path)
        if landmarks is None: return img_orig, False

        img_np = np.array(img_orig)
        perturbed_np = img_orig.copy()
        
        mask_pil = Image.new('L', img_orig.size, 0)
        draw = ImageDraw.Draw(mask_pil)

        groups_to_perturb = []
        if perturb_expressive_only: # For video-driven
            groups_to_perturb = [
                list(range(36,42)), # Left eye
                list(range(42,48)), # Right eye
                list(range(48,68)), # Mouth
            ]
        elif perturb_specific_group: # For attribute edit
            # Pick one or two groups randomly
            all_groups = [
                list(range(17,22)), # Left eyebrow
                list(range(22,27)), # Right eyebrow
                list(range(27,36)), # Nose
                list(range(36,42)), # Left eye
                list(range(42,48)), # Right eye
                list(range(48,68)), # Mouth
            ]
            num_groups = random.randint(2,5)
            groups_to_perturb = random.sample(all_groups, num_groups)
        else: # Full face perturbation (original behavior)
            # Using convex hull of all landmarks
            all_landmarks_np = np.array(landmarks)
            hull_indices = cv2.convexHull(all_landmarks_np.astype(np.float32), returnPoints=False)
            if hull_indices is not None and len(hull_indices) > 0:
                face_hull = [tuple(map(int, pt)) for pt in all_landmarks_np[hull_indices.flatten()]]
                draw.polygon(face_hull, fill=255)


        for group_indices in groups_to_perturb:
            group_pts = [landmarks[i] for i in group_indices]
            if len(group_pts) > 2:
                # Expand polygon slightly
                np_pts = np.array(group_pts)
                center = np.mean(np_pts, axis=0)
                scaled_pts = center + 1.5 * (np_pts - center) # Expand margin even more
                # Ensure coordinates are a list of tuples of Python ints
                scaled_pts_int = [tuple(map(int, pt)) for pt in scaled_pts.tolist()]
                draw.polygon(scaled_pts_int, fill=255)
        
        mask_np = np.array(mask_pil)

        # 1. Elastic deformation (as in [66])
        def elastic_deform(mask, alpha, sigma):
            """Apply elastic deformation to a mask."""
            random_state = np.random.RandomState(None)
            shape = mask.shape
            dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            deformed = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return deformed

        # Typical elastic deformation parameters (tune as needed)
        alpha = random.uniform(5, 15)  # deformation intensity
        sigma = random.uniform(6, 10)  # smoothness
        mask_np = elastic_deform(mask_np, alpha, sigma)

        # 2. Smooth by two Gaussian filters with different parameters
        ksize1 = random.choice([7, 9, 11, 13, 15])
        ksize2 = random.choice([3, 5, 7])
        # Ensure odd kernel sizes
        ksize1 = ksize1 if ksize1 % 2 == 1 else ksize1 + 1
        ksize2 = ksize2 if ksize2 % 2 == 1 else ksize2 + 1

        mask_np = cv2.GaussianBlur(mask_np, (ksize1, ksize1), 0)
        # After first smoothing, set pixels < 1 to 0 (erode/dilate effect)
        mask_np[mask_np < 1] = 0
        mask_np = cv2.GaussianBlur(mask_np, (ksize2, ksize2), 0)

        # 3. Vary blending ratio r ∈ {0.25, 0.5, 0.75, 1, 1, 1}
        r = random.choice([0.25, 0.5, 0.75, 1, 1, 1])
        mask_np = (mask_np * r).clip(0, 255).astype(np.uint8)
        # Apply perturbation only to masked area
        if np.any(mask_np):
            mask = Image.fromarray(mask_np, 'L')
            # Apply perturbation only to masked area
            perturbed_img = self.SBI_transform_pil(perturbed_np)
            perturbed_np = Image.composite(perturbed_img, img_orig, mask)
            return perturbed_np, True
        else:
            return img_orig, False 
        
    def _self_blended_image(self, img, path, stronger_inner_aug=False, subtle_blend=False):
        img_orig = img.copy()
        landmarks = self._get_landmarks(img_orig, path)
        if landmarks is None: return img_orig, False

        # 1. Create initial mask using convex hull of landmarks
        mask = Image.new('L', img.size, 0)
        draw = ImageDraw.Draw(mask)
        all_landmarks_np = np.array(landmarks)
        hull_indices = cv2.convexHull(all_landmarks_np.astype(np.float32), returnPoints=False)
        if hull_indices is None or len(hull_indices) == 0: return img_orig, False
        face_hull = [tuple(map(int, pt)) for pt in all_landmarks_np[hull_indices.flatten()]]
        draw.polygon(face_hull, fill=255)
        mask_np = np.array(mask).astype(np.float32) / 255.0

        # 2. Elastic deformation (as in [66])
        def elastic_deform_mask(mask_np, alpha=img.size[0]*2, sigma=img.size[0]*0.08):
            # mask_np: float32, [0,1], shape (H,W)
            random_state = np.random.RandomState(None)
            shape = mask_np.shape
            dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            deformed = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            return deformed

        mask_np = elastic_deform_mask(mask_np)

        # 3. Smooth mask with two Gaussian filters with different parameters
        # Randomly choose kernel sizes for erosion/dilation effect
        k1 = random.choice([7, 9, 11, 13])
        k2 = random.choice([3, 5, 7])
        if k1 < k2:
            k1, k2 = k2, k1  # Ensure k1 >= k2 for possible erosion

        mask_np1 = cv2.GaussianBlur(mask_np, (k1, k1), 0)
        # After first smoothing, set pixels < 1 to 0 (erode)
        mask_np1 = np.where(mask_np1 < 1.0, 0.0, mask_np1)
        mask_np2 = cv2.GaussianBlur(mask_np1, (k2, k2), 0)

        # 4. Vary blending ratio by multiplying mask by r ∈ {0.25, 0.5, 0.75, 1, 1, 1}
        r = random.choice([0.25, 0.5, 0.75, 1, 1, 1])
        mask_np2 = np.clip(mask_np2 * r, 0, 1)

        # Convert back to PIL mask
        mask = Image.fromarray((mask_np2 * 255).astype(np.uint8), mode='L')

        # Create shifted/augmented inner content
        # Shift slightly
        dx, dy = random.randint(-5,5), random.randint(-5,5)
        # PIL's transform is a bit tricky for simple shifts, use affine
        shifted_img = img_orig.transform(img_orig.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BICUBIC)

        shifted_img = self.SBI_transform_pil(shifted_img) # Apply strong augs
        
        if random.random() < 0.5:
            # add gaussian noise
            noise = np.random.randint(0, 50, size=img_orig.size + (3,), dtype=np.uint8)
            noise_map = Image.fromarray(noise, 'RGB')
            noise_map = noise_map.filter(ImageFilter.GaussianBlur(radius=random.uniform(5,15)))
            shifted_img = Image.blend(shifted_img, noise_map, alpha=random.uniform(0.05, 0.1))
            
        # 네, mask 값(255인 부분)에 따라 shifted_img와 img_orig를 섞는 코드입니다.
        # mask는 alpha(투명도) 마스크처럼 동작합니다. mask 값이 255(불투명)인 영역은 shifted_img가, 0(투명)인 영역은 img_orig가 사용되며,
        # 중간값(예: 128 등)이 있으면 두 이미지가 alpha blending되어 자연스럽게 섞입니다.
        blended = Image.composite(shifted_img, img_orig, mask)
        return blended, True

    def _texture_disturbance(self, img: Image.Image, path, local_effect=False, subtle=False) -> (Image.Image, bool):
        return img, False
        
    def _outlier_distortion(self, img: Image.Image, path) -> (Image.Image, bool):
        return img, False
        
    def _downsample_and_pad(self, img, factor, pad_color='gray'):
        original_width, original_height = img.size
        
        new_width = max(1, original_width // factor) # Ensure at least 1 pixel
        new_height = max(1, original_height // factor)

        downsampled = img.resize((new_width, new_height), Image.BILINEAR)
        
        pad_val = (128,128,128) if pad_color == 'gray' else (0,0,0)
        padded_img = Image.new('RGB', (original_width, original_height), pad_val)
        
        paste_x = (original_width - new_width) // 2
        paste_y = (original_height - new_height) // 2
        padded_img.paste(downsampled, (paste_x, paste_y))
        return padded_img

# Example Usage (requires dummy image files and predictor model)
if __name__ == '__main__':
    print("Running FASDataset example...")
    
    # Create dummy root dir and protocol file for testing
    dummy_root = "dummy_fas_data"
    os.makedirs(os.path.join(dummy_root, "live"), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, "spoof"), exist_ok=True)
    
    # Create dummy images if they don't exist
    live_img_path = os.path.join(dummy_root, "live", "sample_live1.png")
    spoof_img_path = os.path.join(dummy_root, "spoof", "sample_spoof1.png")

    if not os.path.exists(live_img_path):
        Image.new('RGB', (256, 256), color = 'green').save(live_img_path)
        print(f"Created dummy live image: {live_img_path}")
    if not os.path.exists(spoof_img_path):
        Image.new('RGB', (256, 256), color = 'red').save(spoof_img_path)
        print(f"Created dummy spoof image: {spoof_img_path}")

    # Check for dlib model
    if not os.path.exists(SHAPE_PREDICTOR_PATH):
        print(f"FATAL: Dlib shape predictor model '{SHAPE_PREDICTOR_PATH}' not found. Download and place it correctly.")
    else:
        dummy_protocol_path = os.path.join(dummy_root, "protocol.txt")
        with open(dummy_protocol_path, 'w') as f:
            # Relative paths from dummy_root
            f.write(f"live/sample_live1.png 0_0_0 0\n") # Assuming 0_0_0 for live
            f.write(f"spoof/sample_spoof1.png 1_0_0 1\n") # Example spoof
        print(f"Created dummy protocol file: {dummy_protocol_path}")

        # Test dataset initialization
        try:
            fas_dataset = FASDataset(protocol_file=dummy_protocol_path, fas_root_dir=dummy_root, pseudo_rate=1.0) # pseudo_rate 1.0 for testing all pseudo types
            
            if len(fas_dataset) > 0:
                print(f"Dataset initialized with {len(fas_dataset)} samples.")
                
                # Test getting a few items
                for i in range(min(5, len(fas_dataset))):
                    img_tensor, label = fas_dataset[i]
                    print(f"Sample {i}: tensor shape: {img_tensor.shape}, label: {label.item()}")
                    # To visualize, you'd convert tensor back to PIL Image
                    # unloader = transforms.ToPILImage()
                    # img_pil = unloader(img_tensor.squeeze(0))
                    # img_pil.save(f"test_output_sample_{i}_label_{label.item()}.png")
                    # print(f"Saved test_output_sample_{i}_label_{label.item()}.png")

                # Test specific pseudo-attacks on a live image
                print("\nTesting specific pseudo-attacks generation (will save images):")
                live_img_pil = Image.open(live_img_path).convert('RGB')
                live_img_pil = live_img_pil.resize((256,256)) # Resize to expected input for pseudo functions

                all_attack_codes = FASDataset.PHYSICAL_CODES + FASDataset.DIGITAL_CODES
                for code in all_attack_codes:
                    print(f"  Generating pseudo-attack for code: {code}")
                    pseudo_img, success = fas_dataset._apply_subtype_transform(live_img_pil.copy(), code, live_img_path)
                    if success:
                        pseudo_img.save(f"pseudo_attack_test_{code}.png")
                        print(f"    Saved pseudo_attack_test_{code}.png")
                    else:
                        print(f"    Failed to generate pseudo-attack for {code} (e.g. landmark detection failed).")

            else:
                print("Dataset is empty after initialization. Check dummy file paths and parse_protocol_file.")

        except FileNotFoundError as e:
            print(f"Error during dataset test: {e}. Likely dlib model is missing.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()