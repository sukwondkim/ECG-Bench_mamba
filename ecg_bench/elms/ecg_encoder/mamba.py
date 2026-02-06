import torch
from torch import nn
from mamba_ssm import Mamba2 as Mamba
from collections import namedtuple

CombinedOutput = namedtuple("CombinedOutput", ["loss", "out"])


class MambaPretrain(nn.Module):
    def __init__(self, lm, d_model=512, d_state=16, expand=2, seq_len=2500, num_leads=12, **kwargs):
        super(MambaPretrain, self).__init__()
        
        self.input_projection = nn.Linear(num_leads, d_model)
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand,
            )
            for _ in range(4)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(4)
        ])
        
        # MAE-style masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_ratio = kwargs.get('mask_ratio', 0.5)
        self.block_size = kwargs.get('block_size', 25)
        self.masking_strategy = kwargs.get('masking_strategy', 'block')
        
        # Simple decoder for MAE-style reconstruction
        self.decoder_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2) 
            for _ in range(1)
        ])
        self.decoder_layer_norms = nn.LayerNorm(d_model)
        self.decoder_head = nn.Linear(d_model, num_leads)

    def block_masking(self, x, mask_ratio, block_size):
        """
        Block-wise masking: Mask blocks to prevent linear interpolation
        
        Note: When seq_len is not divisible by block_size, the remaining timesteps
        at the end are not masked.

        Args:
            x: [B, seq_len, num_leads] - original ECG signals (before projection)
            mask_ratio: ratio of timesteps to mask
            block_size: size of each block to mask
        
        Returns:
            x_masked: [B, len_keep, num_leads] - only visible tokens
            mask: [B, seq_len] - binary mask (0=keep, 1=remove)
            ids_restore: [B, seq_len] - indices to restore original order
        """
        B, L, D = x.shape
        
        num_blocks = L // block_size
        num_masked_blocks = int(num_blocks * mask_ratio)
        
        # Create mask at block level
        mask = torch.zeros(B, L, device=x.device)
        
        if num_blocks > 0 and num_masked_blocks > 0:
            # Randomly select blocks to mask
            block_noise = torch.rand(B, num_blocks, device=x.device)
            block_indices = torch.argsort(block_noise, dim=1)
            
            masked_block_indices = block_indices[:, :num_masked_blocks]  # [B, num_masked_blocks]

            masked_token_indices = masked_block_indices.unsqueeze(-1) * block_size + torch.arange(block_size, device=x.device).view(1, 1, -1)
            masked_token_indices = masked_token_indices.reshape(B, -1)  # [B, num_masked_blocks * block_size]
            mask.scatter_(1, masked_token_indices, 1)
            
        len_keep = L - (num_masked_blocks * block_size)
        
        sort_key = mask * (L + 1) + torch.arange(L, device=x.device).unsqueeze(0)
        ids_shuffle = torch.argsort(sort_key, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        return x_masked, mask, ids_restore
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking (original MAE-style)
        
        Args:
            x: [B, seq_len, num_leads] - original ECG signals (before projection)

        Returns:
            x_masked: [B, len_keep, num_leads] - only visible tokens
            mask: [B, seq_len] - binary mask (0=keep, 1=remove)
            ids_restore: [B, seq_len] - indices to restore original order
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def encode_ecg(self, ecg_signals):
        """Encode ECG signals using Mamba blocks with block-wise or random masking
        
        Args:
            ecg_signals: [B, num_leads, seq_len] - original ECG signals

        Returns:
            ecg_features: [B, len_keep, d_model] - encoded ECG features after masking
            mask: [B, seq_len] - binary mask (0=keep, 1=remove)
            ids_restore: [B, seq_len] - indices to restore original order
        """
        # ecg_signals: [B, num_leads, seq_len] -> [B, seq_len, num_leads]
        ecg_signals = ecg_signals.transpose(1, 2)
        
        # Apply masking based on strategy (before projection)
        if self.masking_strategy == 'block':
            ecg_signals, mask, ids_restore = self.block_masking(ecg_signals, self.mask_ratio, self.block_size)
        else:
            ecg_signals, mask, ids_restore = self.random_masking(ecg_signals, self.mask_ratio)
        
        x = self.input_projection(ecg_signals)
        for mamba_block, layer_norm in zip(self.mamba_blocks, self.layer_norms):
            x = mamba_block(x)
            x = layer_norm(x)
        
        return x, mask, ids_restore
    
    def decode_ecg(self, ecg_features, ids_restore):
        """Decode ECG features back to original signal space for reconstruction
        
        Args:
            ecg_features: [B, len_keep, d_model] - encoded ECG features after masking
            ids_restore: [B, seq_len] - indices to restore original order
        
        Returns:
            reconstructed: [B, num_leads, seq_len] - reconstructed ECG signals
        """
        B, _, D = ecg_features.shape
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - ecg_features.shape[1], 1)
        x_full = torch.cat([ecg_features, mask_tokens], dim=1)
        
        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Apply decoder blocks
        for decoder_block in self.decoder_blocks:
            x_full = decoder_block(x_full)
        x_full = self.decoder_layer_norms(x_full)
        reconstructed = self.decoder_head(x_full)

        reconstructed = reconstructed.transpose(1, 2)
        
        return reconstructed
    
    def compute_masked_loss(self, original, reconstructed, mask):
        """
        Compute MSE loss only on masked patches (like MAE)
        
        Args:
            original: [B, num_leads, seq_len] - original ECG signals
            reconstructed: [B, num_leads, seq_len] - prediction
            mask: [B, seq_len], 0 is keep, 1 is remove
        
        Returns:
            loss: scalar - masked MSE loss
        """
        # Expand mask to match num_leads dimension
        mask = mask.unsqueeze(1)  # [B, 1, seq_len]
        
        # Compute loss: [B, num_leads, seq_len]
        loss = (reconstructed - original) ** 2
        
        # Mean loss per timestep (average across leads, like MAE averages across patch pixels)
        loss = loss.mean(dim=1)  # [B, seq_len]
        
        # Mean loss on masked timesteps only
        loss = (loss * mask.squeeze(1)).sum() / mask.sum()
        
        return loss
    
    def forward(self, batch):
        ecg_signals = batch["ecg_signal"]
        ecg_signals = ecg_signals.to(self.input_projection.weight.dtype)

        x, mask, ids_restore = self.encode_ecg(ecg_signals)
        reconstructed = self.decode_ecg(x, ids_restore)
        
        recon_loss = self.compute_masked_loss(ecg_signals, reconstructed, mask)

        return CombinedOutput(loss=recon_loss, out=reconstructed)


class MambaFinetune(nn.Module):
    """Mamba encoder for Stage 2 (LLaVA training with LLM)"""
    def __init__(self, d_model=512, d_state=16, expand=2, num_leads=12, num_encoder_tokens=1, **kwargs):
        super(MambaFinetune, self).__init__()
        
        # ECG encoder: Mamba blocks
        self.input_projection = nn.Linear(num_leads, d_model)
        
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=expand
            )
            for _ in range(4)  # 4 Mamba layers
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(4)
        ])
        
        self.avgpool = nn.AdaptiveAvgPool1d(num_encoder_tokens)

    def forward(self, batch):
        ecg_signals = batch["ecg_signal"]
        ecg_signals = ecg_signals.to(self.input_projection.weight.dtype)
        ecg_signals = ecg_signals.transpose(1, 2)  # [B, seq_len, num_leads]
        x = self.input_projection(ecg_signals)
        
        for mamba_block, layer_norm in zip(self.mamba_blocks, self.layer_norms):
            x = mamba_block(x)
            x = layer_norm(x)
        
        return CombinedOutput(loss=None, out=x)
    
    @torch.no_grad()
    def get_encoder_embeddings(self, batch):
        ecg_signals = batch["ecg_signal"]
        ecg_signals = ecg_signals.to(self.input_projection.weight.dtype)
        ecg_signals = ecg_signals.transpose(1, 2)  # [B, seq_len, num_leads]
        x = self.input_projection(ecg_signals)
        
        for mamba_block, layer_norm in zip(self.mamba_blocks, self.layer_norms):
            x = mamba_block(x)
            x = layer_norm(x)
        
        x = x.transpose(1, 2)  # [B, d_model, seq_len]
        x = self.avgpool(x)     # [B, d_model, num_encoder_tokens]
        x = x.transpose(1, 2)   # [B, num_encoder_tokens, d_model]
        
        return x