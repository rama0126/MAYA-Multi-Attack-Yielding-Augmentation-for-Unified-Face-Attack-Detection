# /workspace/FAS_ICCV/train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score 
import argparse
import os
import numpy as np
import time # For saving path if needed
import logging

from fas_datasets import FASDataset
from models.classifier import SwinFASClassifier # Assuming models.classifier exists
from utils import calculate_acer
from sam import SAM # Import the SAM optimizer

# Setup logger
logger = logging.getLogger("FAS_Train")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Path configurations
DEFAULT_FAS_ROOT = "/workspace/datasets/FAS"
DEFAULT_FULL_TRAIN_PROTOCOL = os.path.join(DEFAULT_FAS_ROOT, "phase1/Protocol-train.txt")
DEFAULT_CLASSIFIER_SAVE_DIR = "/workspace/FAS_ICCV/classifier_checkpoints_kfold_sam" # Changed save dir name
DEFAULT_MAE_ENCODER_PATH = "/workspace/FAS_ICCV/mae_checkpoints/swin_mae_encoder_final.pth"

# Validation set txts
DEFAULT_CDF_VAL_TXT = "/workspace/DFIL/DFIL/dataset_txts/CDF_val.txt"
DEFAULT_FFIW_VAL_TXT = "/workspace/DFIL/DFIL/dataset_txts/FFIW_val.txt"

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, log_interval, use_sam, label_smoothing_epsilon=0.0): # Added label_smoothing_epsilon
    model.train()
    total_train_loss = 0
    train_preds_probs, train_targets_original = [], [] # Store original hard targets for metrics

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        # labels are initially 0 or 1
        original_labels = labels.to(device).unsqueeze(1).float() # Keep original for metrics

        # Apply label smoothing
        if label_smoothing_epsilon > 0:
            target_ones_value = 1.0 - label_smoothing_epsilon
            target_zeros_value = label_smoothing_epsilon
            
            smoothed_labels = torch.empty_like(original_labels)
            smoothed_labels[original_labels == 1] = target_ones_value
            smoothed_labels[original_labels == 0] = target_zeros_value
            
            target_for_loss = smoothed_labels
        else:
            target_for_loss = original_labels
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, target_for_loss) # Use smoothed labels for loss calculation
        loss.backward()

        if use_sam:
            optimizer.first_step(zero_grad=True)
            # Use smoothed labels for the second backward pass as well
            criterion(model(images), target_for_loss).backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        total_train_loss += loss.item() 
        
        with torch.no_grad():
            train_preds_probs.extend(torch.sigmoid(logits).cpu().numpy())
        # For metrics like accuracy and AUC, use the original hard labels
        train_targets_original.extend(original_labels.cpu().numpy())

        if (batch_idx + 1) % log_interval == 0:
            logger.info(f"Epoch [{epoch+1}/{total_epochs}], Batch [{batch_idx+1}/{len(loader)}], Train Loss: {loss.item():.4f}")
    
    avg_train_loss = total_train_loss / len(loader)
    # Metrics are calculated against original hard labels
    train_targets_original_np = np.array(train_targets_original).flatten()
    train_preds_probs_np = np.array(train_preds_probs).flatten()
    
    train_accuracy = accuracy_score(train_targets_original_np, train_preds_probs_np > 0.5)
    try:
        train_auc = roc_auc_score(train_targets_original_np, train_preds_probs_np)
    except ValueError:
        train_auc = 0.0
        
    return avg_train_loss, train_accuracy, train_auc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_val_loss = 0
    val_preds_probs, val_targets = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Ensure labels are float32 for BCEWithLogitsLoss and compatible with model output
            labels = labels.to(device).unsqueeze(1).float()
            logits = model(images)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            
            val_preds_probs.extend(torch.sigmoid(logits).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(loader)
    val_targets_np = np.array(val_targets).flatten()
    val_preds_probs_np = np.array(val_preds_probs).flatten()

    apcer, bpcer, acer = calculate_acer(val_targets_np, val_preds_probs_np, threshold=0.5)
    val_accuracy = accuracy_score(val_targets_np, val_preds_probs_np > 0.5)
    try:
        val_auc = roc_auc_score(val_targets_np, val_preds_probs_np)
    except ValueError:
        val_auc = 0.0

    return avg_val_loss, val_accuracy, val_auc, apcer, bpcer, acer

def validate_on_dataset(model, dataset, criterion, device, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return validate_one_epoch(model, loader, criterion, device)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"Using device: {device}")

    # Add run_name to save_dir for unique directories
    if args.run_name:
        args.save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # Standard transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from utils import parse_protocol_file as parse_protocol_util

    # Load train set
    all_parsed_samples = parse_protocol_util(args.full_train_protocol, args.fas_root_dir)
    all_dataset_samples = []
    for full_path, attack_code, binary_label in all_parsed_samples:
        if binary_label is not None:
            all_dataset_samples.append((full_path, attack_code, binary_label))
        else:
            logger.warning(f"Sample {full_path} has no binary label. Skipping.")

    if not all_dataset_samples:
        logger.error(f"No valid samples from {args.full_train_protocol}. Exiting.")
        return

    # Load validation set from two txts, and keep them separate for per-domain evaluation
    cdf_val_samples = []
    ffiw_val_samples = []
    # 이런 형식이야
    # label,path
    # 0,/workspace/datasets/deepfakes/FFIW/source/face_images/val/val_00000000/0013.png
    with open(args.cdf_val_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, path = line.split(",", 1)
            cdf_val_samples.append((path, None, int(label)))
    with open(args.ffiw_val_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, path = line.split(",", 1)
            ffiw_val_samples.append((path, None, int(label)))
    if not cdf_val_samples and not ffiw_val_samples:
        logger.error(f"No valid samples from validation txts. Exiting.")
        return

    # For backward compatibility, also keep the merged val set (for best_acer selection)
    val_samples = cdf_val_samples + ffiw_val_samples

    train_dataset = FASDataset(samples_list=all_dataset_samples, transform=train_transform)
    val_dataset = FASDataset(samples_list=val_samples, transform=val_transform)
    cdf_val_dataset = FASDataset(samples_list=cdf_val_samples, transform=val_transform)
    ffiw_val_dataset = FASDataset(samples_list=ffiw_val_samples, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SwinFASClassifier(
        model_name=args.model_name,
        num_classes=1,
        img_size=args.img_size,
        pretrained_encoder_path=args.mae_encoder_path,
        freeze_encoder=args.freeze_encoder
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer setup
    optimizer_params = []
    if not args.freeze_encoder and args.mae_encoder_path:
        optimizer_params.append({'params': model.encoder.parameters(), 'lr': args.lr * 0.1})
        optimizer_params.append({'params': model.classifier_head.parameters(), 'lr': args.lr})
    else:
        optimizer_params = model.parameters()

    if args.use_sam:
        logger.info("Using SAM optimizer.")
        if not args.freeze_encoder and args.mae_encoder_path:
            optimizer = SAM(optimizer_params, torch.optim.AdamW, rho=args.sam_rho, adaptive=args.sam_adaptive, lr=args.lr, weight_decay=args.weight_decay)
        else: # freeze_encoder or no pretrained_encoder
            optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=args.sam_rho, adaptive=args.sam_adaptive, lr=args.lr, weight_decay=args.weight_decay)
    else:
        logger.info("Using AdamW optimizer.")
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
        
    # Scheduler should operate on the base_optimizer if SAM is used, or the optimizer itself.
    opt_for_scheduler = optimizer.base_optimizer if args.use_sam else optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_for_scheduler, T_max=args.epochs, eta_min=args.lr * 0.01 if args.lr > 0 else 0)

    best_acer = float('inf')
    best_model_path = ""

    for epoch in range(args.epochs):
        avg_train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, 
            args.log_interval, args.use_sam, args.label_smoothing # Pass label_smoothing
        )
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] Avg Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, LR: {current_lr:.6e}")

        # Validation on merged set (for best_acer selection)
        avg_val_loss, val_acc, val_auc, apcer, bpcer, acer = validate_one_epoch(
            model, val_loader, criterion, device
        )
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] Avg Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] APCER: {apcer:.4f}, BPCER: {bpcer:.4f}, ACER: {acer:.4f}")

        # Per-domain validation
        if len(cdf_val_samples) > 0:
            cdf_val_loss, cdf_val_acc, cdf_val_auc, cdf_apcer, cdf_bpcer, cdf_acer = validate_on_dataset(
                model, cdf_val_dataset, criterion, device, args.batch_size, args.num_workers
            )
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] [CDF] Val Loss: {cdf_val_loss:.4f}, Acc: {cdf_val_acc:.4f}, AUC: {cdf_val_auc:.4f}, APCER: {cdf_apcer:.4f}, BPCER: {cdf_bpcer:.4f}, ACER: {cdf_acer:.4f}")
        if len(ffiw_val_samples) > 0:
            ffiw_val_loss, ffiw_val_acc, ffiw_val_auc, ffiw_apcer, ffiw_bpcer, ffiw_acer = validate_on_dataset(
                model, ffiw_val_dataset, criterion, device, args.batch_size, args.num_workers
            )
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] [FFIW] Val Loss: {ffiw_val_loss:.4f}, Acc: {ffiw_val_acc:.4f}, AUC: {ffiw_val_auc:.4f}, APCER: {ffiw_apcer:.4f}, BPCER: {ffiw_bpcer:.4f}, ACER: {ffiw_acer:.4f}")
        
        scheduler.step()

        if acer < best_acer:
            best_acer = acer
            # Remove previously saved best model to avoid clutter
            if best_model_path and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except OSError as e:
                    logger.error(f"Error removing old best model {best_model_path}: {e}")
            
            best_model_path = os.path.join(args.save_dir, f"best_model_epoch_{epoch+1}_acer_{acer:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path} (ACER: {acer:.4f})")

    # Evaluate the best model
    if best_model_path and os.path.exists(best_model_path):
        logger.info(f"Loading best model from {best_model_path} for final evaluation.")
        model.load_state_dict(torch.load(best_model_path)) 

        # Evaluate on merged val set
        _, val_acc, val_auc, apcer, bpcer, acer = validate_one_epoch(
            model, val_loader, criterion, device # Re-validate with best weights
        )
        logger.info(f"Final Metrics (from best model, merged val): ACER: {acer:.4f}, APCER: {apcer:.4f}, BPCER: {bpcer:.4f}, ACC: {val_acc:.4f}, AUC: {val_auc:.4f}")

        # Evaluate on CDF
        if len(cdf_val_samples) > 0:
            _, cdf_val_acc, cdf_val_auc, cdf_apcer, cdf_bpcer, cdf_acer = validate_on_dataset(
                model, cdf_val_dataset, criterion, device, args.batch_size, args.num_workers
            )
            logger.info(f"Final Metrics (from best model, CDF): ACER: {cdf_acer:.4f}, APCER: {cdf_apcer:.4f}, BPCER: {cdf_bpcer:.4f}, ACC: {cdf_val_acc:.4f}, AUC: {cdf_val_auc:.4f}")
        # Evaluate on FFIW
        if len(ffiw_val_samples) > 0:
            _, ffiw_val_acc, ffiw_val_auc, ffiw_apcer, ffiw_bpcer, ffiw_acer = validate_on_dataset(
                model, ffiw_val_dataset, criterion, device, args.batch_size, args.num_workers
            )
            logger.info(f"Final Metrics (from best model, FFIW): ACER: {ffiw_acer:.4f}, APCER: {ffiw_apcer:.4f}, BPCER: {ffiw_bpcer:.4f}, ACC: {ffiw_val_acc:.4f}, AUC: {ffiw_val_auc:.4f}")
    else:
        logger.warning("Best model was not saved or found. Skipping metrics.")

    logger.info("Classifier Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swin FAS Classifier Training with SAM (no k-fold, fixed val set)")
    # ... (기존 인자들)
    parser.add_argument('--model_name', type=str, default='swin_tiny_patch4_window7_224')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5) # For base optimizer
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--fas_root_dir', type=str, default=DEFAULT_FAS_ROOT)
    parser.add_argument('--full_train_protocol', type=str, default=DEFAULT_FULL_TRAIN_PROTOCOL)
    parser.add_argument('--mae_encoder_path', type=str, default=DEFAULT_MAE_ENCODER_PATH)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=DEFAULT_CLASSIFIER_SAVE_DIR)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--use_gpu', action='store_true', help="Enable GPU usage if available") # Added use_gpu flag
    parser.add_argument('--run_name', type=str, default=f"fas_train_{time.strftime('%Y%m%d-%H%M')}", help="A name for this run, used for save directory.")

    # Validation txts
    parser.add_argument('--cdf_val_txt', type=str, default=DEFAULT_CDF_VAL_TXT, help="Path to CDF_val.txt")
    parser.add_argument('--ffiw_val_txt', type=str, default=DEFAULT_FFIW_VAL_TXT, help="Path to FFIW_val.txt")

    # SAM specific arguments
    parser.add_argument('--use_sam', action='store_true', help='Use SAM optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05, help='Rho for SAM optimizer')
    parser.add_argument('--sam_adaptive', action='store_true', help='Use adaptive SAM')

    # Label Smoothing argument
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Epsilon for label smoothing (e.g., 0.1). 0.0 means no smoothing.')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.use_gpu: # Check use_gpu flag
        torch.cuda.manual_seed_all(args.seed)
        # Setting deterministic can slow down, consider if truly needed
        # torch.backends.cudnn.deterministic = True 
        # torch.backends.cudnn.benchmark = False
    else: # If GPU not available or not requested, ensure reproducibility on CPU
        pass

    main(args)