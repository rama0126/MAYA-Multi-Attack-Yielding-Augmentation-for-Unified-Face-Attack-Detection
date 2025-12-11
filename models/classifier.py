# /workspace/FAS_ICCV/models_classifier.py
import torch
import torch.nn as nn
import timm

class SwinFASClassifier(nn.Module):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', num_classes=1, img_size=256, pretrained_encoder_path=None, freeze_encoder=False):
        super().__init__()
        # Load Swin Transformer encoder
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size) # num_classes=0 gives features
        
        if pretrained_encoder_path:
            print(f"Loading pretrained MAE encoder weights from: {pretrained_encoder_path}")
            try:
                encoder_state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
                # Adjust keys if needed (e.g. if saved with "module." prefix from DataParallel)
                if all(key.startswith('module.') for key in encoder_state_dict.keys()):
                    encoder_state_dict = {k.replace('module.', ''): v for k, v in encoder_state_dict.items()}
                
                # Load into self.encoder (which is the SwinTransformer instance)
                missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
                print(f"Encoder weights loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                if unexpected_keys: print(f"Unexpected keys: {unexpected_keys}")
                if missing_keys: print(f"Missing keys: {missing_keys}")

            except Exception as e:
                print(f"Error loading pretrained encoder: {e}. Starting with random Swin weights.")
        else:
            print("No pretrained MAE encoder path provided. Using random Swin weights (or timm's default if pretrained=True was set).")

        if freeze_encoder:
            print("Freezing encoder weights.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Get feature dimension from the encoder
        # For Swin, num_features is available after model creation
        self.feature_dim = self.encoder.num_features 
        
        # Classification head
        # For binary classification (Live/Spoof), num_classes=1 with BCEWithLogitsLoss
        # or num_classes=2 with CrossEntropyLoss
        self.dropout = nn.Dropout(0.1)
        self.classifier_head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        # Encoder forward_features typically returns [B, L, C] for ViT/Swin if no pooling
        # Or, if global_pool is part of the model (often is when num_classes=0), it returns [B,C]
        # timm's SwinTransformer with num_classes=0 has `self.head = nn.Identity()`
        # Its `forward` method calls `forward_features` then `self.head`.
        # `forward_features` output is [B, NumPatches, EmbedDim] before norm&pool.
        # After `self.norm` (in `forward_features`) and `self.avgpool` (if global_pool='avg'), then `flatten`
        # The `self.encoder(x)` should give [B, feature_dim]
        features = self.encoder(x) # This should be [B, feature_dim]
        
        # If features are not [B, feature_dim], you might need explicit pooling:
        # e.g. if features = self.encoder.forward_features(x) -> [B, L, C]
        # features = torch.mean(features, dim=1) # Global average pooling over patch sequence
        features = self.dropout(features)
        logits = self.classifier_head(features)
        return logits