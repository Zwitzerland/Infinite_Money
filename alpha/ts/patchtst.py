"""
PatchTST and CT-PatchTST Implementation
Long-context forecasting with channel-time patching for time series.

References:
- "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (arXiv:2211.14730)
- CT-PatchTST: Channel-Time patching variant for improved long horizons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import math

from ...utils.logger import get_logger


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST models."""
    # Input dimensions
    seq_len: int = 512  # Input sequence length
    pred_len: int = 96  # Prediction horizon
    n_vars: int = 7     # Number of variables (multivariate)
    
    # Patch configuration
    patch_len: int = 16  # Patch length
    stride: int = 8      # Patch stride
    
    # Transformer configuration
    d_model: int = 128   # Model dimension
    n_heads: int = 8     # Number of attention heads
    e_layers: int = 6    # Number of encoder layers
    d_ff: int = 256      # Feed-forward dimension
    dropout: float = 0.1 # Dropout rate
    
    # CT-PatchTST specific
    channel_independence: bool = True  # Channel independence
    temporal_resolution: List[int] = None  # Multi-resolution patches
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


class PatchEmbedding(nn.Module):
    """Patch embedding layer for time series."""
    
    def __init__(self, patch_len: int, stride: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Linear projection for patches
        self.proj = nn.Linear(patch_len * in_channels, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, embed_dim))  # Max 1000 patches
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_vars)
        Returns:
            patches: Patch embeddings (batch_size, n_patches, embed_dim)
            n_patches: Number of patches
        """
        B, L, C = x.shape
        
        # Create patches
        n_patches = (L - self.patch_len) // self.stride + 1
        patches = []
        
        for i in range(n_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_len
            patch = x[:, start_idx:end_idx, :]  # (B, patch_len, C)
            patches.append(patch.reshape(B, -1))  # (B, patch_len * C)
        
        patches = torch.stack(patches, dim=1)  # (B, n_patches, patch_len * C)
        
        # Project to embedding dimension
        embedded = self.proj(patches)  # (B, n_patches, embed_dim)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:n_patches].unsqueeze(0)
        
        return embedded, n_patches


class ChannelEmbedding(nn.Module):
    """Channel embedding for multivariate time series."""
    
    def __init__(self, n_vars: int, embed_dim: int):
        super().__init__()
        self.n_vars = n_vars
        self.embed_dim = embed_dim
        
        # Channel embeddings
        self.channel_emb = nn.Embedding(n_vars, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add channel embeddings to patch embeddings."""
        B, n_patches, _ = x.shape
        
        # Create channel indices
        channel_indices = torch.arange(self.n_vars, device=x.device)
        channel_embeddings = self.channel_emb(channel_indices)  # (n_vars, embed_dim)
        
        # Replicate for each patch
        channel_embeddings = channel_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, n_vars, embed_dim)
        channel_embeddings = channel_embeddings.expand(B, n_patches, -1, -1)  # (B, n_patches, n_vars, embed_dim)
        
        return channel_embeddings


class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words
    
    Transformer-based model using patch-based tokenization for long-term forecasting.
    """
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("PatchTST")
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=config.patch_len,
            stride=config.stride,
            in_channels=config.n_vars,
            embed_dim=config.d_model
        )
        
        # Channel embedding
        if not config.channel_independence:
            self.channel_embedding = ChannelEmbedding(config.n_vars, config.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.e_layers)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.pred_len)
        
        # Channel independence handling
        if config.channel_independence:
            # Each channel processed independently
            self.channel_projections = nn.ModuleList([
                nn.Linear(config.d_model, config.pred_len) for _ in range(config.n_vars)
            ])
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for PatchTST.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_vars)
            
        Returns:
            predictions: Forecasted values (batch_size, pred_len, n_vars)
        """
        B, L, C = x.shape
        
        if self.config.channel_independence:
            # Process each channel independently
            predictions = []
            
            for c in range(C):
                # Extract single channel
                x_c = x[:, :, c:c+1]  # (B, L, 1)
                
                # Patch embedding
                patches, n_patches = self.patch_embedding(x_c)  # (B, n_patches, d_model)
                
                # Transformer encoding
                encoded = self.transformer(patches)  # (B, n_patches, d_model)
                
                # Global average pooling
                pooled = encoded.mean(dim=1)  # (B, d_model)
                
                # Output projection
                pred_c = self.channel_projections[c](pooled)  # (B, pred_len)
                predictions.append(pred_c.unsqueeze(-1))
            
            predictions = torch.cat(predictions, dim=-1)  # (B, pred_len, n_vars)
            
        else:
            # Process all channels together
            # Patch embedding
            patches, n_patches = self.patch_embedding(x)  # (B, n_patches, d_model)
            
            # Add channel information
            channel_emb = self.channel_embedding(patches)  # (B, n_patches, n_vars, d_model)
            
            # Reshape for transformer
            patches = patches.unsqueeze(2).expand(-1, -1, C, -1)  # (B, n_patches, n_vars, d_model)
            patches = patches + channel_emb
            patches = patches.reshape(B, n_patches * C, -1)  # (B, n_patches * n_vars, d_model)
            
            # Transformer encoding
            encoded = self.transformer(patches)  # (B, n_patches * n_vars, d_model)
            
            # Reshape back
            encoded = encoded.reshape(B, n_patches, C, -1)  # (B, n_patches, n_vars, d_model)
            
            # Global average pooling over patches
            pooled = encoded.mean(dim=1)  # (B, n_vars, d_model)
            
            # Output projection
            predictions = []
            for c in range(C):
                pred_c = self.output_projection(pooled[:, c, :])  # (B, pred_len)
                predictions.append(pred_c.unsqueeze(-1))
            
            predictions = torch.cat(predictions, dim=-1)  # (B, pred_len, n_vars)
        
        return predictions


class CTSimplePatchTST(nn.Module):
    """
    CT-PatchTST: Channel-Time patching variant
    
    Improves channel-time patching for longer horizons with multi-resolution processing.
    """
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("CT-PatchTST")
        
        # Multi-resolution patch lengths
        if config.temporal_resolution is None:
            self.patch_lengths = [8, 16, 32]  # Multi-scale patches
        else:
            self.patch_lengths = config.temporal_resolution
        
        # Multiple patch embeddings for different resolutions
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(
                patch_len=patch_len,
                stride=patch_len // 2,
                in_channels=config.n_vars,
                embed_dim=config.d_model
            ) for patch_len in self.patch_lengths
        ])
        
        # Channel-Time attention
        self.channel_time_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Transformer encoders for each resolution
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    batch_first=True
                ),
                num_layers=config.e_layers // len(self.patch_lengths)
            ) for _ in self.patch_lengths
        ])
        
        # Cross-scale fusion
        self.scale_fusion = nn.Linear(len(self.patch_lengths) * config.d_model, config.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.pred_len * config.n_vars)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CT-PatchTST.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_vars)
            
        Returns:
            predictions: Forecasted values (batch_size, pred_len, n_vars)
        """
        B, L, C = x.shape
        
        # Multi-resolution processing
        scale_features = []
        
        for i, (patch_emb, transformer) in enumerate(zip(self.patch_embeddings, self.transformers)):
            # Patch embedding at current resolution
            patches, n_patches = patch_emb(x)  # (B, n_patches, d_model)
            
            # Transformer encoding
            encoded = transformer(patches)  # (B, n_patches, d_model)
            
            # Global average pooling
            pooled = encoded.mean(dim=1)  # (B, d_model)
            scale_features.append(pooled)
        
        # Cross-scale fusion
        fused = torch.cat(scale_features, dim=-1)  # (B, len(patch_lengths) * d_model)
        fused = self.scale_fusion(fused)  # (B, d_model)
        fused = self.dropout(fused)
        
        # Channel-Time attention
        fused_expanded = fused.unsqueeze(1).expand(-1, C, -1)  # (B, n_vars, d_model)
        attended, _ = self.channel_time_attention(
            fused_expanded, fused_expanded, fused_expanded
        )  # (B, n_vars, d_model)
        
        # Global pooling over channels
        pooled = attended.mean(dim=1)  # (B, d_model)
        
        # Output projection
        output = self.output_projection(pooled)  # (B, pred_len * n_vars)
        predictions = output.reshape(B, self.config.pred_len, C)  # (B, pred_len, n_vars)
        
        return predictions


class PatchTSTPredictor:
    """
    High-level interface for PatchTST and CT-PatchTST models.
    """
    
    def __init__(self, config: PatchTSTConfig, model_type: str = "patchtst"):
        """
        Initialize PatchTST predictor.
        
        Args:
            config: Model configuration
            model_type: Either "patchtst" or "ct-patchtst"
        """
        self.config = config
        self.model_type = model_type
        self.logger = get_logger(f"PatchTSTPredictor_{model_type}")
        
        # Initialize model
        if model_type == "patchtst":
            self.model = PatchTST(config)
        elif model_type == "ct-patchtst":
            self.model = CTSimplePatchTST(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step."""
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(x)
        loss = self.criterion(predictions, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        x = x.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(x)
        
        return predictions.cpu()
    
    def forecast_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate forecasts with uncertainty quantification.
        
        Args:
            x: Input sequence
            n_samples: Number of Monte Carlo samples
            
        Returns:
            mean_forecast: Mean prediction
            uncertainty: Standard deviation of predictions
        """
        x = x.to(self.device)
        self.model.train()  # Enable dropout for uncertainty
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x)
                predictions.append(pred.cpu())
        
        predictions = torch.stack(predictions)  # (n_samples, batch_size, pred_len, n_vars)
        
        mean_forecast = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_forecast, uncertainty
    
    def get_calibrated_quantiles(self, x: torch.Tensor, quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict[float, torch.Tensor]:
        """
        Get calibrated quantile predictions.
        
        Args:
            x: Input sequence
            quantiles: List of quantiles to compute
            
        Returns:
            quantile_predictions: Dictionary mapping quantiles to predictions
        """
        mean_forecast, uncertainty = self.forecast_with_uncertainty(x)
        
        quantile_predictions = {}
        for q in quantiles:
            if q == 0.5:
                quantile_predictions[q] = mean_forecast
            else:
                # Assume Gaussian distribution for simplicity
                from scipy.stats import norm
                z_score = norm.ppf(q)
                quantile_predictions[q] = mean_forecast + z_score * uncertainty
        
        return quantile_predictions