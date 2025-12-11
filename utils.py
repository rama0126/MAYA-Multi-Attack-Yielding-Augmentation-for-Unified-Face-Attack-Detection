# /workspace/FAS_ICCV/models_mae.py

# Positional encoding (from MAE official repo)
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import numpy as np
import os
from sklearn.metrics import confusion_matrix

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to prepare embeddings for
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def parse_protocol_file(protocol_file_path, data_root_path, phase="train"):
    """
    Parses the protocol file and returns a list of (image_path, label_str, binary_label) tuples.
    For phase 2 (classification), it also returns binary labels (0 for live, 1 for spoof).
    """
    samples = []
    with open(protocol_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            relative_image_path = parts[0]
            # Ensure the relative_image_path correctly maps to the data_root_path structure
            # e.g. if relative_image_path is "Data-train/image.png" and data_root_path is "/workspace/datasets/FAS"
            # then full_image_path should be "/workspace/datasets/FAS/Data-train/image.png"
            # If data_root_path is "/workspace/datasets/FAS/Data-train", then just join with os.path.basename(relative_image_path)
            # Based on your initial description, data_root_path is like "/workspace/datasets/FAS/Data-train"
            # and relative_image_path is "Data-train/20445.png".
            # The parse_protocol_file is called with image_data_dir which is ALREADY Data-train or Data-val
            # So, relative_image_path should be just "20445.png" if image_data_dir is "/workspace/datasets/FAS/Data-train"
            # Let's adjust this based on the given example: "Data-train/20445.png" is the relative path from FAS root
            
            # The current dataset class calls it with image_data_dir = /workspace/datasets/FAS/Data-train (or Data-val)
            # And the protocol file contains "Data-train/20445.png".
            # So, data_root_path for parse_protocol_file should be "/workspace/datasets/FAS"
            # And then we append the relative_image_path.
            # Let's rename data_root_path argument to dataset_base_dir to reflect this.
            full_image_path = os.path.join(data_root_path, relative_image_path) # This was `dataset_base_dir` before, assuming it's `/workspace/datasets/FAS`
                                                                              # If `data_root_path` is passed as `/workspace/datasets/FAS/Data-train`,
                                                                              # then `relative_image_path` should be just the filename `20445.png`.
                                                                              # Let's stick to the initial interpretation:
                                                                              # protocol has "Data-train/img.png"
                                                                              # image_data_dir for FASDataset is "/workspace/datasets/FAS/Data-train"
                                                                              # This means parse_protocol_file should be called with the *overall dataset root*
                                                                              # The FASDataset currently calls it with `image_data_dir`.
                                                                              # This means if protocol has "Data-train/img.png" and image_data_dir is "/ws/ds/FAS/Data-train",
                                                                              # then full_path becomes "/ws/ds/FAS/Data-train/Data-train/img.png" which is wrong.

            # Correction for path joining in parse_protocol_file:
            # The protocol file paths are relative to the FAS directory.
            # e.g. "Data-train/20445.png"
            # The `image_data_dir` passed to FASDataset is e.g. "/workspace/datasets/FAS/Data-train"
            # So, the `parse_protocol_file` needs the FAS root.
            # Let's assume `data_root_path` in `parse_protocol_file` is the *actual directory where images are*,
            # and the protocol file path component (like "Data-train/") is just for label association.
            # The simplest is: protocol file lists paths *relative to the dataset_base_dir*
            # So if protocol has "image_name.png" and image_data_dir is "Data-train", then path is "Data-train/image_name.png"
            #
            # Your format:
            # Protocol: "Data-train/20445.png 2_1_0"
            # image_data_dir in FASDataset: "/workspace/datasets/FAS/Data-train"
            # This implies parse_protocol_file should take `fas_root_dir` instead of `image_data_dir`
            # and `image_data_dir` is only for filtering.
            # OR, parse_protocol_file should adjust the path from the protocol.
            #
            # Let's assume:
            # `protocol_file_path` refers to files like `/workspace/datasets/FAS/phase1/Protocol-train.txt`
            # `data_root_path` for `parse_protocol_file` will be `/workspace/datasets/FAS` (the main root)
            # The `relative_image_path` from protocol is e.g. `Data-train/20445.png`
            # So `full_image_path` = `os.path.join("/workspace/datasets/FAS", "Data-train/20445.png")`
            # This seems correct. The FASDataset will need to pass this main root.

            if len(parts) > 1: # Label is present
                label_str = parts[1]
                binary_label = 0 if label_str == "0_0_0" else 1
                samples.append((full_image_path, label_str, binary_label))
            else: 
                samples.append((full_image_path, None, None)) 
                
    return samples


def get_live_samples(parsed_samples):
    """Filters parsed samples to return only live ones for MAE pre-training."""
    live_samples = []
    for full_image_path, label_str, _ in parsed_samples:
        if label_str == "0_0_0":
            live_samples.append(full_image_path)
    return live_samples

def calculate_acer(y_true, y_pred_probs, threshold=0.5):
    """
    Calculates APCER, BPCER, and ACER.
    Assumes:
        y_true: 0 for Live, 1 for Spoof
        y_pred_probs: Sigmoid output probabilities for Spoof class
        threshold: Threshold to convert probabilities to binary predictions
    """
    y_pred_binary = (y_pred_probs > threshold).astype(int)

    # Confusion matrix:
    # TN: True Negative (Actual Live, Predicted Live)
    # FP: False Positive (Actual Live, Predicted Spoof)
    # FN: False Negative (Actual Spoof, Predicted Live)
    # TP: True Positive (Actual Spoof, Predicted Spoof)
    #       Predicted
    #       0 (Live)  1 (Spoof)
    # Actual 0 (Live)   TN        FP
    #        1 (Spoof)  FN        TP
    
    # Ensure y_true and y_pred_binary are 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred_binary = np.array(y_pred_binary).flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

    # APCER: Attack Presentation Classification Error Rate
    # Spoof samples classified as Live / Total Spoof samples
    # FN / (FN + TP)
    if (fn + tp) == 0:
        apcer = 0.0  # Or handle as an error/warning, e.g., if no spoof samples in evaluation
        print("Warning: No spoof samples found for APCER calculation (FN+TP=0). APCER set to 0.")
    else:
        apcer = fn / (fn + tp)

    # BPCER: Bona Fide Presentation Classification Error Rate
    # Live samples classified as Spoof / Total Live samples
    # FP / (FP + TN)
    if (fp + tn) == 0:
        bpcer = 0.0  # Or handle as an error/warning, e.g., if no live samples
        print("Warning: No live samples found for BPCER calculation (FP+TN=0). BPCER set to 0.")
    else:
        bpcer = fp / (fp + tn)
        
    acer = (apcer + bpcer) / 2.0
    
    return apcer, bpcer, acer