"""
TimeGrad and Multi-Resolution Diffusion Models
Calibrated predictive densities for time series forecasting.

References:
- "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting" (ICML 2021)
- "Multi-Resolution Diffusion Models for Time Series Forecasting" (ICLR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

from ...utils.logger import get_logger


@dataclass
class DiffusionConfig:
    """Configuration for diffusion models."""
    # Data dimensions
    input_dim: int = 7       # Number of input features
    output_dim: int = 7      # Number of output features
    context_length: int = 256 # Context length
    prediction_length: int = 64 # Prediction length
    
    # Diffusion parameters
    num_steps: int = 100     # Number of diffusion steps
    beta_start: float = 1e-4 # Starting beta value
    beta_end: float = 0.02   # Ending beta value
    beta_schedule: str = "linear"  # Beta schedule type
    
    # Model architecture
    residual_layers: int = 36    # Number of residual layers
    residual_channels: int = 64  # Residual channels
    dilation_cycle_length: int = 10  # Dilation cycle length
    
    # Multi-resolution
    resolution_levels: List[int] = None  # Different resolution levels
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract values from tensor a at indices t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule for diffusion."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear beta schedule for diffusion."""
    return torch.linspace(beta_start, beta_end, timesteps)


class ResidualBlock(nn.Module):
    """Residual block for diffusion model."""
    
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        
        # Dilated convolution
        self.conv_dilated = nn.Conv1d(
            channels, channels * 2, kernel_size=3, 
            padding=dilation, dilation=dilation
        )
        
        # Point-wise convolution
        self.conv_pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Skip connection
        self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, channels, length)
            
        Returns:
            residual: Residual output
            skip: Skip connection output
        """
        residual = x
        
        # Layer norm (transpose for correct dimensions)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        
        # Dilated convolution
        x = self.conv_dilated(x)
        
        # Split into gate and filter
        gate, filter_val = torch.chunk(x, 2, dim=1)
        x = torch.sigmoid(gate) * torch.tanh(filter_val)
        
        # Point-wise convolution
        x = self.conv_pointwise(x)
        
        # Skip connection
        skip = self.conv_skip(x)
        
        # Residual connection
        residual = x + residual
        
        return residual, skip


class TimeEmbedding(nn.Module):
    """Time embedding for diffusion steps."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Time embedding MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate time embeddings."""
        # Sinusoidal embeddings
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.mlp(emb)


class TimeGradModel(nn.Module):
    """
    TimeGrad: Autoregressive Denoising Diffusion Model for Time Series.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("TimeGrad")
        
        # Input projection
        self.input_projection = nn.Conv1d(
            config.input_dim, config.residual_channels, kernel_size=1
        )
        
        # Time embedding
        self.time_embedding = TimeEmbedding(config.residual_channels)
        
        # Residual layers
        self.residual_layers = nn.ModuleList()
        for i in range(config.residual_layers):
            dilation = 2 ** (i % config.dilation_cycle_length)
            self.residual_layers.append(
                ResidualBlock(config.residual_channels, dilation)
            )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Conv1d(config.residual_channels, config.residual_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(config.residual_channels, config.output_dim, kernel_size=1)
        )
        
        # Context encoder
        self.context_encoder = nn.LSTM(
            config.input_dim, config.residual_channels // 2,
            batch_first=True, bidirectional=True
        )
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for TimeGrad.
        
        Args:
            x: Noisy input (batch_size, output_dim, prediction_length)
            timesteps: Diffusion timesteps (batch_size,)
            context: Context sequence (batch_size, context_length, input_dim)
            
        Returns:
            noise_pred: Predicted noise (batch_size, output_dim, prediction_length)
        """
        # Time embedding
        time_emb = self.time_embedding(timesteps)  # (batch_size, residual_channels)
        
        # Context encoding
        context_encoded, _ = self.context_encoder(context)  # (batch_size, context_length, residual_channels)
        context_pooled = context_encoded.mean(dim=1)  # (batch_size, residual_channels)
        
        # Combine time and context embeddings
        cond_emb = time_emb + context_pooled  # (batch_size, residual_channels)
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, residual_channels, prediction_length)
        
        # Add conditioning
        cond_emb = cond_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x + cond_emb
        
        # Residual layers
        skip_connections = []
        for layer in self.residual_layers:
            x, skip = layer(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = torch.stack(skip_connections).sum(dim=0)
        
        # Output projection
        noise_pred = self.output_projection(x)
        
        return noise_pred


class MultiResolutionDiffusion(nn.Module):
    """
    Multi-Resolution Diffusion Model for Time Series.
    
    Processes different temporal resolutions to capture multi-scale patterns.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.logger = get_logger("MultiResDiffusion")
        
        # Resolution levels
        if config.resolution_levels is None:
            self.resolution_levels = [1, 2, 4, 8]  # Different downsampling factors
        else:
            self.resolution_levels = config.resolution_levels
        
        # Separate models for each resolution
        self.resolution_models = nn.ModuleDict()
        for res in self.resolution_levels:
            self.resolution_models[str(res)] = TimeGradModel(config)
        
        # Cross-resolution fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(len(self.resolution_levels) * config.output_dim, config.output_dim * 2),
            nn.ReLU(),
            nn.Linear(config.output_dim * 2, config.output_dim)
        )
        
    def downsample(self, x: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample tensor by given factor."""
        if factor == 1:
            return x
        
        # Average pooling for downsampling
        kernel_size = min(factor, x.shape[-1])
        pooled = F.avg_pool1d(x, kernel_size=kernel_size, stride=factor)
        
        return pooled
    
    def upsample(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Upsample tensor to target length."""
        if x.shape[-1] == target_length:
            return x
        
        # Linear interpolation for upsampling
        upsampled = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        
        return upsampled
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-resolution diffusion.
        
        Args:
            x: Noisy input (batch_size, output_dim, prediction_length)
            timesteps: Diffusion timesteps (batch_size,)
            context: Context sequence (batch_size, context_length, input_dim)
            
        Returns:
            noise_pred: Predicted noise (batch_size, output_dim, prediction_length)
        """
        target_length = x.shape[-1]
        resolution_outputs = []
        
        # Process each resolution
        for res in self.resolution_levels:
            # Downsample input
            x_res = self.downsample(x, res)
            context_res = self.downsample(context.transpose(1, 2), res).transpose(1, 2)
            
            # Process with resolution-specific model
            noise_pred_res = self.resolution_models[str(res)](x_res, timesteps, context_res)
            
            # Upsample back to original resolution
            noise_pred_res = self.upsample(noise_pred_res, target_length)
            
            resolution_outputs.append(noise_pred_res)
        
        # Fuse multi-resolution outputs
        concatenated = torch.cat(resolution_outputs, dim=1)  # (batch_size, output_dim * num_res, prediction_length)
        concatenated = concatenated.transpose(1, 2)  # (batch_size, prediction_length, output_dim * num_res)
        
        fused = self.fusion_network(concatenated)  # (batch_size, prediction_length, output_dim)
        fused = fused.transpose(1, 2)  # (batch_size, output_dim, prediction_length)
        
        return fused


class DiffusionPredictor:
    """
    High-level interface for diffusion-based time series forecasting.
    """
    
    def __init__(self, config: DiffusionConfig, model_type: str = "timegrad"):
        """
        Initialize diffusion predictor.
        
        Args:
            config: Model configuration
            model_type: Either "timegrad" or "multi-resolution"
        """
        self.config = config
        self.model_type = model_type
        self.logger = get_logger(f"DiffusionPredictor_{model_type}")
        
        # Initialize model
        if model_type == "timegrad":
            self.model = TimeGradModel(config)
        elif model_type == "multi-resolution":
            self.model = MultiResolutionDiffusion(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize diffusion schedule
        self._setup_diffusion_schedule()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def _setup_diffusion_schedule(self):
        """Setup diffusion noise schedule."""
        if self.config.beta_schedule == "linear":
            betas = linear_beta_schedule(
                self.config.num_steps,
                self.config.beta_start,
                self.config.beta_end
            )
        elif self.config.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.config.num_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor.to(self.device))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion process (add noise)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def train_step(self, x: torch.Tensor, context: torch.Tensor) -> float:
        """Single training step."""
        x, context = x.to(self.device), context.to(self.device)
        
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.config.num_steps, (batch_size,), device=self.device).long()
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Forward diffusion
        x_noisy = self.q_sample(x, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t, context)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Single reverse diffusion step."""
        # Predict noise
        predicted_noise = self.model(x, t, context)
        
        # Extract coefficients
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        # Predict x_0
        x_0_pred = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
        
        return x_0_pred
    
    @torch.no_grad()
    def sample(self, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate samples using reverse diffusion."""
        context = context.to(self.device)
        batch_size = context.shape[0]
        
        # Start from pure noise
        shape = (batch_size * num_samples, self.config.output_dim, self.config.prediction_length)
        x = torch.randn(shape, device=self.device)
        
        # Expand context for multiple samples
        context_expanded = context.repeat_interleave(num_samples, dim=0)
        
        # Reverse diffusion
        for t in reversed(range(self.config.num_steps)):
            t_tensor = torch.full((batch_size * num_samples,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, context_expanded)
        
        # Reshape to separate samples
        x = x.view(batch_size, num_samples, self.config.output_dim, self.config.prediction_length)
        
        return x
    
    def predict_probabilistic(self, context: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts.
        
        Args:
            context: Context sequence
            num_samples: Number of samples to generate
            
        Returns:
            forecasts: Dictionary with mean, std, and quantiles
        """
        self.model.eval()
        
        # Generate samples
        samples = self.sample(context, num_samples)  # (batch_size, num_samples, output_dim, prediction_length)
        
        # Transpose for easier computation
        samples = samples.transpose(2, 3)  # (batch_size, num_samples, prediction_length, output_dim)
        
        # Compute statistics
        mean_forecast = samples.mean(dim=1)  # (batch_size, prediction_length, output_dim)
        std_forecast = samples.std(dim=1)   # (batch_size, prediction_length, output_dim)
        
        # Compute quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        quantile_forecasts = {}
        for q in quantiles:
            quantile_forecasts[f"q{int(q*100)}"] = torch.quantile(samples, q, dim=1)
        
        return {
            "mean": mean_forecast,
            "std": std_forecast,
            "samples": samples,
            **quantile_forecasts
        }