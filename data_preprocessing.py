# data_preprocessing.py (Final Version with ImportError fix)

import os
import cv2
import numpy as np
import argparse
import joblib
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# --- SOLUTION: Import directly from the source file to avoid circular import ---
from insightface.app.face_analysis import FaceAnalysis

# face_parsing 폴더가 프로젝트 루트에 있다고 가정
from face_parsing.mask_model import BiSeNet
from torchvision import transforms

class FaceImageDataset(Dataset):
    def __init__(self, image_paths, root_dir, transform=None):
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        img_rel_path = self.image_paths[index]
        img_full_path = os.path.join(self.root_dir, img_rel_path)
        try:
            img = Image.open(img_full_path).convert("RGB")
            shape = np.array(img).shape
        except Exception:
            return img_rel_path, torch.zeros(3, 512, 512), (0,0,0)

        image_resized = img.resize((512, 512), Image.BILINEAR)

        # BiSeNet 입력에 맞는 Transform
        bisenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        if self.transform:
            image_resized = self.transform(image_resized)
        
        return img_rel_path, image_resized, shape

    def __len__(self):
        return len(self.image_paths)


def crop_face_and_save(img_info, input_dir, output_dir, face_detector, confidence_thresh, delta):
    try:
        rel_path = img_info['rel_path']
        input_path = os.path.join(input_dir, rel_path)
        output_path = os.path.join(output_dir, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        img = cv2.imread(input_path)
        if img is None:
            return False, img_info

        height, width, _ = img.shape
        faces = face_detector.get(img)

        valid_faces = [f for f in faces if f['det_score'] > confidence_thresh] if faces else []

        if valid_faces:
            best_face = max(valid_faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
            bbox = [int(b) for b in best_face['bbox']]
            pt1 = [max(0, bbox[0] - delta), max(0, bbox[1] - delta)]
            pt2 = [min(width, bbox[2] + delta), min(height, bbox[3] + delta)]
            
            if pt1[1] < pt2[1] and pt1[0] < pt2[0]:
                cropped_face = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                    cv2.imwrite(output_path, cropped_face)
                    return True, img_info
        
        cv2.imwrite(output_path, img)
        return False, img_info

    except Exception as e:
        print(f"Error processing {img_info.get('rel_path', 'unknown')}: {e}")
        return False, img_info


def process_split(args, split_name, face_detector, mask_net=None):
    print(f"\n{'='*20} Processing '{split_name}' data {'='*20}")

    base_dir = args.base_dir
    input_data_dir = base_dir
    metadata_path = os.path.join(base_dir, 'phase1', f'Protocol-{split_name}.txt')
    output_base_dir = os.path.join(base_dir, f'Data-{split_name}_preprocessed')
    output_image_dir = os.path.join(output_base_dir, 'images')
    os.makedirs(output_image_dir, exist_ok=True)
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}. Skipping '{split_name}' split.")
        return

    all_image_info_initial = []
    live_label_code = '0_0_0'
    live_binary_label = 1
    fake_binary_label = 0

    with open(metadata_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            rel_path = parts[0]
            info = {'rel_path': rel_path}
            if split_name == 'train' or split_name == 'val':
                label_str = parts[1]
                is_live = (label_str == live_label_code)
                info['binary_label'] = live_binary_label if is_live else fake_binary_label
                info['code'] = label_str
            else:
                info['binary_label'] = -1
            all_image_info_initial.append(info)
    print(f"Found {len(all_image_info_initial)} images initially for '{split_name}'.")

    face_detected_info = []
    print(f"--- Cropping faces for '{split_name}' ---")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(crop_face_and_save, info, input_data_dir, output_image_dir, face_detector, args.det_thresh, args.det_delta): info 
            for info in all_image_info_initial
        }
        pbar = tqdm.tqdm(as_completed(futures), total=len(all_image_info_initial), desc=f"Processing {split_name} Images")
        for future in pbar:
            detected, info = future.result()
            if detected:
                face_detected_info.append(info)

    print(f"Face detection successful for {len(face_detected_info)} / {len(all_image_info_initial)} images.")
    
    if split_name == 'train' and mask_net is not None:
        live_face_detected_paths = [info['rel_path'] for info in face_detected_info if info['binary_label'] == live_binary_label]
        
        if live_face_detected_paths:
            print(f"--- Generating masks for {len(live_face_detected_paths)} LIVE images with detected faces ---")
            output_mask_dir = os.path.join(output_base_dir, 'masks')
            os.makedirs(output_mask_dir, exist_ok=True)
            for rel_path in live_face_detected_paths:
                img_subdir = os.path.dirname(rel_path)
                os.makedirs(os.path.join(output_mask_dir, img_subdir), exist_ok=True)
            
            bisenet_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            mask_dataset = FaceImageDataset(live_face_detected_paths, output_image_dir, transform=bisenet_transform)
            mask_loader = DataLoader(mask_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            device = next(mask_net.parameters()).device

            with torch.no_grad():
                for rel_paths, resized_imgs_tensor, original_shapes_batch in tqdm.tqdm(mask_loader, desc="Generating Masks"):
                    resized_imgs_tensor = resized_imgs_tensor.to(device)
                    out = mask_net(resized_imgs_tensor)[0]
                    masks = out.argmax(1).cpu().numpy().astype(np.uint8)
                    # print(original_shapes_batch)
                    for i, mask in enumerate(masks):
                        original_h = original_shapes_batch[0][i].item()
                        original_w = original_shapes_batch[1][i].item()
                        if original_h == 0 or original_w == 0: continue
                        
                        rel_path = rel_paths[i]
                        img_name = os.path.basename(rel_path).split('.')[0]
                        img_subdir = os.path.dirname(rel_path)
                        mask_save_path = os.path.join(output_mask_dir, img_subdir, img_name + '.pkl')
                        
                        mask_resized = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                        joblib.dump(mask_resized, mask_save_path)
        else:
            print("--- No LIVE images with detected faces to generate masks for. ---")

    if split_name == 'train' or split_name == 'val':
        if split_name == 'train':
            final_list_path = os.path.join(output_base_dir, f'{split_name}_label.txt')
        else:
            final_list_path = os.path.join(output_base_dir, f'{split_name}_list.txt')
        with open(final_list_path, 'w') as f:
            for info in all_image_info_initial:
                image_output_rel_path = os.path.join('images', info['rel_path'])
                # binary_label과 attack_code를 함께 저장
                f.write(f"{image_output_rel_path} {info['binary_label']} {info['code']}\n")
    else: # val
        final_list_path = os.path.join(output_base_dir,  f'{split_name}_list.txt')
        with open(final_list_path, 'w') as f:
            for info in all_image_info_initial:
                image_output_rel_path = os.path.join('images', info['rel_path'])
                f.write(f"{image_output_rel_path}\n")

    print(f"List file for '{split_name}' created at: {final_list_path}")


def main(args):
    print("Initializing models...")
    face_detector = FaceAnalysis(
        name='buffalo_l', 
        allowed_modules=['detection']
    )
    face_detector.prepare(
        ctx_id=args.ctx_id, 
        det_size=(args.det_size, args.det_size)
    )
    
    mask_net = None
    if os.path.exists(args.mask_model_path):
        device = torch.device(f"cuda:{args.ctx_id}" if args.ctx_id >= 0 else "cpu")
        n_classes = 19
        mask_net = BiSeNet(n_classes=n_classes)
        mask_net.load_state_dict(torch.load(args.mask_model_path, map_location='cpu'))
        mask_net = torch.nn.DataParallel(mask_net).to(device)
        mask_net.eval()
        print("Mask generation model loaded successfully.")
    else:
        print(f"Warning: Mask model not found at {args.mask_model_path}. Skipping all mask generation.")

    process_split(args, 'train', face_detector, mask_net)
    process_split(args, 'val', face_detector, mask_net=None)
    process_split(args, 'test', face_detector, mask_net=None)

    print("\nAll preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Face Attack Detection - Data Preprocessing")
    
    parser.add_argument('--base_dir', type=str, default='/workspace/datasets/FAS2', help='Dataset base directory.')
    parser.add_argument('--mask_model_path', type=str, default='face_parsing/79999_iter.pth', help='Path to the pretrained BiSeNet model.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for parallel processing.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for mask generation.')
    parser.add_argument('--ctx_id', type=int, default=0, help='GPU device id to use. -1 for CPU.')
    parser.add_argument('--det_size', type=int, default=320, help='Input size for face detector.')
    parser.add_argument('--det_thresh', type=float, default=0.3, help='Confidence threshold for face detection.')
    parser.add_argument('--det_delta', type=int, default=20, help='Pixel margin to expand the detected face bbox.')

    args = parser.parse_args()

    if args.ctx_id < 0 and args.num_workers > 1:
        print("Warning: Using multiple workers on CPU might be inefficient. Consider setting --num_workers 1.")

    main(args)