# /workspace/FAS_ICCV/models_mae.py
import torch
import torch.nn as nn
import timm
import numpy as np # For positional encoding

# Positional encoding (from MAE official repo, kept for completeness)
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
# --------------------------------------------------------

class SimpleSwinMAE(nn.Module):
    def __init__(self, model_name='swin_large_patch4_window7_224', img_size=256, 
                 user_patch_size=32, # New argument for user-defined patch size for reconstruction
                 mask_ratio=0.75, decoder_dim=512, decoder_depth=2):
        super().__init__()
        self.img_size = img_size
        self.reconstruction_patch_size = user_patch_size # User-defined patch size for the target
        self.mask_ratio = mask_ratio

        self.swin_downsample_factor = 32 # Swin's typical total downsampling

        # Encoder setup
        _encoder_temp = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        self.encoder_output_dim = _encoder_temp.num_features
        self.num_tokens_from_encoder = (img_size // self.swin_downsample_factor)**2 # e.g., (256/32)^2 = 64
        
        # print(f"DEBUG: Model: {model_name}, Img Size: {img_size}, User Patch Size: {self.reconstruction_patch_size}")
        # print(f"DEBUG: Encoder Output Dim: {self.encoder_output_dim}")
        # print(f"DEBUG: Num Tokens from Swin Encoder (L_encoded): {self.num_tokens_from_encoder}")

        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=img_size)
        
        # MAE Decoder
        self.decoder_embed = nn.Linear(self.encoder_output_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens_from_encoder, decoder_dim), requires_grad=False 
        ) 
        
        decoder_blocks = []
        for _ in range(decoder_depth):
            decoder_blocks.append(nn.TransformerEncoderLayer(
                d_model=decoder_dim, 
                nhead=max(1, decoder_dim // 64), 
                dim_feedforward=decoder_dim * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True 
            ))
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        # Decoder predicts pixels for a patch of size `reconstruction_patch_size`
        self.decoder_pred = nn.Linear(decoder_dim, self.reconstruction_patch_size**2 * 3, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # ... (same as before, ensures decoder_pos_embed uses self.num_tokens_from_encoder)
        nn.init.normal_(self.mask_token, std=.02)
        decoder_pos_embed_data = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.num_tokens_from_encoder**0.5),
            cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed_data).float().unsqueeze(0))

        # Initialize decoder layers
        for m in self.decoder_blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        if self.decoder_embed.bias is not None:
            nn.init.zeros_(self.decoder_embed.bias)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            nn.init.zeros_(self.decoder_pred.bias)

    def random_masking(self, x_tokens, mask_ratio):
        # ... (same as before, operates on self.num_tokens_from_encoder tokens)
        N, L, D = x_tokens.shape 
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_unmasked_tokens = torch.gather(x_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x_tokens.device) 
        mask[:, :len_keep] = 0 
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_unmasked_tokens, mask, ids_restore

    def forward_encoder(self, x_imgs):
        # ... (same as before, potentially reshapes Swin output to [N, L_encoded, D])
        encoded_feature_map = self.encoder.forward_features(x_imgs)
        if encoded_feature_map.ndim == 4:
            N, H_feat, W_feat, D_encoded = encoded_feature_map.shape
            assert H_feat * W_feat == self.num_tokens_from_encoder
            assert D_encoded == self.encoder_output_dim
            encoded_tokens = encoded_feature_map.reshape(N, H_feat * W_feat, D_encoded)
        elif encoded_feature_map.ndim == 3:
            encoded_tokens = encoded_feature_map
            N, L_encoded, D_encoded = encoded_tokens.shape
            assert L_encoded == self.num_tokens_from_encoder
            assert D_encoded == self.encoder_output_dim
        else:
            raise ValueError(f"encoded_feature_map has unexpected ndim: {encoded_feature_map.ndim}")
        
        latent_unmasked, mask, ids_restore = self.random_masking(encoded_tokens, self.mask_ratio)
        return latent_unmasked, mask, ids_restore

    def forward_decoder(self, latent_unmasked, ids_restore):
        # ... (same as before)
        x_visible_embed = self.decoder_embed(latent_unmasked)
        N, L_keep, D_decoder = x_visible_embed.shape
        L_total_encoded = ids_restore.shape[1]
        assert L_total_encoded == self.num_tokens_from_encoder
        num_masked = L_total_encoded - L_keep
        mask_tokens = self.mask_token.repeat(N, num_masked, 1)
        x_embed_shuffled = torch.cat([x_visible_embed, mask_tokens], dim=1)
        x_embed_ordered = torch.gather(x_embed_shuffled, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D_decoder))
        x = x_embed_ordered + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred_pixels = self.decoder_pred(x)
        return pred_pixels

    def patchify_target(self, imgs):
        """
        imgs: (N, 3, H, W)
        Returns target patches for MAE loss: (N, L_encoded, target_patch_size**2 * 3)
        where L_encoded = (H/target_patch_size) * (W/target_patch_size)
        and target_patch_size is self.target_patch_size (e.g., 32)
        """
        p_target = self.target_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p_target == 0
        assert imgs.shape[2] // p_target == int(self.num_tokens_from_encoder**0.5)

        h_patch_count = w_patch_count = imgs.shape[2] // p_target # e.g., 256/32 = 8
        
        # Reshape to [N, C, h_patch_count, p_target, w_patch_count, p_target]
        x = imgs.reshape(shape=(imgs.shape[0], 3, h_patch_count, p_target, w_patch_count, p_target))
        # Permute to [N, h_patch_count, w_patch_count, p_target, p_target, C]
        x = torch.einsum('nchpwq->nhwpqc', x)
        # Reshape to [N, L_encoded, p_target*p_target*C]
        # L_encoded = h_patch_count * w_patch_count
        target_patches = x.reshape(shape=(imgs.shape[0], h_patch_count * w_patch_count, p_target**2 * 3))
        assert target_patches.shape[1] == self.num_tokens_from_encoder
        return target_patches

    def unpatchify_output(self, pred_pixel_values):
        """
        pred_pixel_values: (N, L_encoded, reconstruction_patch_size**2 * 3)
        Returns: (N, 3, H_recon, W_recon)
        H_recon = sqrt(L_encoded) * reconstruction_patch_size
        W_recon = sqrt(L_encoded) * reconstruction_patch_size
        """
        N, L_encoded, _ = pred_pixel_values.shape
        assert L_encoded == self.num_tokens_from_encoder

        p_recon = self.reconstruction_patch_size
        h_patch_count = w_patch_count = int(L_encoded**0.5)

        x = pred_pixel_values.reshape(
            shape=(N, h_patch_count, w_patch_count, p_recon, p_recon, 3)
        )
        x = torch.einsum('nhwpqc->nchpwq', x)
        reconstructed_imgs = x.reshape(
            shape=(N, 3, h_patch_count * p_recon, w_patch_count * p_recon)
        )
        return reconstructed_imgs
    
    def forward_loss(self, imgs, pred_pixels, mask):
        # ... (same as before, but target comes from the modified patchify_target)
        target = self.patchify_target(imgs) 
        assert pred_pixels.shape[1] == target.shape[1] == mask.shape[1], \
            f"Shape mismatch: pred {pred_pixels.shape}, target {target.shape}, mask {mask.shape}"
        loss = (pred_pixels - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss

    def forward(self, imgs):
        # ... (same as before)
        latent_unmasked, mask, ids_restore = self.forward_encoder(imgs)
        pred_pixel_values = self.forward_decoder(latent_unmasked, ids_restore)
        loss = self.forward_loss(imgs, pred_pixel_values, mask)
        return loss, pred_pixel_values, mask, ids_restore
    def patchify_target(self, imgs):
        """
        imgs: (N, 3, H, W) original images
        Returns target patches: (N, L_encoded, reconstruction_patch_size**2 * 3)
        It reshapes regions of the original image to match the L_encoded tokens,
        where each region is effectively reconstruction_patch_size x reconstruction_patch_size.
        This requires H and W to be divisible by (sqrt(L_encoded) * reconstruction_patch_size)
        No, this should divide the image into L_encoded "super-patches", and each super-patch
        is then represented as if it were `reconstruction_patch_size` x `reconstruction_patch_size`.
        This is only meaningful if `reconstruction_patch_size` is the size of these super-patches.
        i.e., `reconstruction_patch_size == self.img_size // int(self.num_tokens_from_encoder**0.5)`
        """
        # The size of the region in the original image that each of the L_encoded tokens represents.
        actual_region_size = self.img_size // int(self.num_tokens_from_encoder**0.5) # e.g., 256 / 8 = 32

        # If user_patch_size is different from actual_region_size, we need to resample/crop each region.
        # For simplicity, let's assume user wants to reconstruct these actual_region_size blocks,
        # but the `decoder_pred` output dimension is based on `reconstruction_patch_size`.
        # This means the loss calculation needs care.
        #
        # Simplest interpretation for now: the "target" patch for loss calculation
        # will be made of `reconstruction_patch_size` x `reconstruction_patch_size` pixels,
        # and there will be `self.num_tokens_from_encoder` such target patches.
        # This implies the total image area covered by these target patches is
        # `self.num_tokens_from_encoder * (self.reconstruction_patch_size**2)`.
        # This might not match the original image size if `reconstruction_patch_size` is arbitrary.

        # Let's make `patchify_target` create patches of `self.reconstruction_patch_size`
        # by taking `self.num_tokens_from_encoder` chunks from the image.
        # This implicitly means the image is first downscaled/resampled to
        # `sqrt(num_tokens) * recon_patch_size` by `sqrt(num_tokens) * recon_patch_size`.
        # Or, we take `num_tokens_from_encoder` non-overlapping patches of `recon_patch_size` from the original image.
        # This would mean `img_size` must be `sqrt(num_tokens) * recon_patch_size`.

        p_target = self.reconstruction_patch_size
        num_patches_sqrt = int(self.num_tokens_from_encoder**0.5) # e.g., 8
        
        # Required image size for this patchification: num_patches_sqrt * p_target
        expected_h = expected_w = num_patches_sqrt * p_target
        
        if imgs.shape[2] != expected_h or imgs.shape[3] != expected_w:
            # If image size doesn't match, resample it to the expected size for patchification
            # This means the MAE is learning to reconstruct a resampled version if patch_size is not 32 (for 256 img)
            # print(f"Warning: Original image size {imgs.shape[2:]} differs from expected { (expected_h, expected_w) } for patch_size {p_target}. Resampling input to patchify_target.")
            imgs_resampled = transforms.functional.resize(imgs, [expected_h, expected_w], antialias=True)
        else:
            imgs_resampled = imgs

        h_patch_count = w_patch_count = num_patches_sqrt # This is fixed by Swin encoder output
        
        x = imgs_resampled.reshape(shape=(imgs_resampled.shape[0], 3, h_patch_count, p_target, w_patch_count, p_target))
        x = torch.einsum('nchpwq->nhwpqc', x)
        target_patches = x.reshape(shape=(imgs_resampled.shape[0], h_patch_count * w_patch_count, p_target**2 * 3))
        assert target_patches.shape[1] == self.num_tokens_from_encoder
        return target_patches