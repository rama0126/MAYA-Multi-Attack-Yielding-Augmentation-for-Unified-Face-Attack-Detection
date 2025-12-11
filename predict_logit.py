# /workspace/FAS_ICCV/predict_logits.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from PIL import Image
import numpy as np

# Assuming your models and datasets are in these locations relative to this script
# Adjust import paths if necessary
from models.classifier import SwinFASClassifier # Phase 2 Classifier model
# FASDataset can be simplified for prediction if labels are not needed from protocol file
# For now, let's create a simpler Dataset for inference

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths_list, fas_root_dir, transform=None):
        """
        Args:
            image_paths_list (list): List of relative image paths (e.g., "Data-val/00000.png").
            fas_root_dir (str): Root directory of the FAS dataset (e.g., /workspace/datasets/FAS).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.fas_root_dir = fas_root_dir
        self.image_relative_paths = image_paths_list
        self.transform = transform if transform else self.get_default_transform()

    def get_default_transform(self):
        # Use the same transformations as validation during training, minus augmentations
        return transforms.Compose([
            transforms.Resize((224, 224)), # Should match the img_size used for training classifier
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_relative_paths)

    def __getitem__(self, idx):
        relative_img_path = self.image_relative_paths[idx]
        full_img_path = os.path.join(self.fas_root_dir, relative_img_path)
        
        try:
            image = Image.open(full_img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found at {full_img_path}")
            # Return a dummy tensor or handle appropriately
            # For now, let's return a tensor of zeros and the path, so it can be skipped or noted
            return torch.zeros((3, 224, 224)), relative_img_path # Match transform output size

        if self.transform:
            image = self.transform(image)
        
        return image, relative_img_path # Return image and its relative path

def parse_val_protocol(protocol_file_path):
    """ Parses the validation protocol file to get a list of relative image paths. """
    image_relative_paths = []
    with open(protocol_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            image_relative_paths.append(parts[0]) # Only need the first part (path)
    return image_relative_paths

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(args.model_weights):
        print(f"Error: Model weights not found at {args.model_weights}")
        return
    if not os.path.isfile(args.val_protocol):
        print(f"Error: Validation protocol file not found at {args.val_protocol}")
        return
    if not os.path.isdir(args.fas_root_dir):
        print(f"Error: FAS root directory not found at {args.fas_root_dir}")
        return

    # Load the trained classifier model
    # Ensure num_classes=1 for BCEWithLogitsLoss (outputting a single logit)
    model = SwinFASClassifier(
        model_name=args.model_name, 
        num_classes=1, 
        pretrained_encoder_path=None, # Encoder weights are part of the full classifier state_dict
        img_size=args.img_size
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_weights, map_location=device))
        print(f"Successfully loaded model weights from {args.model_weights}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval() # Set the model to evaluation mode

    # Prepare dataset and dataloader
    image_relative_paths = parse_val_protocol(args.val_protocol)
    if not image_relative_paths:
        print("No image paths found in the protocol file.")
        return

    # Use the img_size that the classifier was trained with
    # This should be consistent with the SwinFASClassifier's expected input size
    # For SwinFASClassifier, it's likely 224, but we can make it an arg
    inference_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = InferenceDataset(
        image_paths_list=image_relative_paths,
        fas_root_dir=args.fas_root_dir,
        transform=inference_transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    results = [] # To store (relative_path, logit)

    with torch.no_grad(): # Disable gradient calculations for inference
        from tqdm import tqdm

        for batch_idx, (images, rel_paths_batch) in enumerate(tqdm(dataloader, desc="Predicting", unit="batch")):
            images = images.to(device)
            
            logits = model(images) # Output shape [batch_size, 1]
            logits = torch.sigmoid(logits)
            logits = 1 - logits
            # Ensure logits are on CPU and converted to numpy for easier handling
            logits_np = logits.cpu().numpy().flatten() # Flatten to get a 1D array of logits
            
            for rel_path, logit_val in zip(rel_paths_batch, logits_np):
                if isinstance(rel_path, torch.Tensor): # Handle dummy path from FileNotFoundError
                    if torch.all(images[0] == 0): # Crude check for dummy image
                        print(f"Skipping dummy entry for a likely missing file: {rel_path}")
                        continue
                results.append((rel_path, logit_val))

    # Write results to output file
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_file, 'w') as f:
        for rel_path, logit_val in results:
            f.write(f"{rel_path} {logit_val:.5f}\n") # Format logit to 5 decimal places, adjust as needed
            
    print(f"Predictions saved to {args.output_file}")
    print(f"Total predictions made: {len(results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Logits for FAS Validation Set")
    
    # Paths
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the trained classifier model weights (.pth file)')
    parser.add_argument('--val_protocol', type=str, default='/workspace/datasets/FAS/phase1/Protocol-val.txt', help='Path to the Protocol-val.txt file')
    parser.add_argument('--fas_root_dir', type=str, default="/workspace/datasets/FAS", help='Root directory of the FAS dataset (e.g., /workspace/datasets/FAS)')
    parser.add_argument('--output_file', type=str, default='./val_predictions.txt', help='Path to save the output predictions')

    # Model and Data Config
    parser.add_argument('--model_name', type=str, default='swin_large_patch4_window7_224', help='Name of the Swin Transformer model used for the classifier (must match trained model architecture)')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size for the model (must match training)')
    
    # Execution Config
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval (batches) for logging processing status')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available (default is to use GPU if available, this flag is somewhat redundant but can be explicit)')

    args = parser.parse_args()
    main(args)