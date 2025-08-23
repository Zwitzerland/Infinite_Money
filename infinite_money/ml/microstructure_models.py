"""
Microstructure Edge Models
Implements DeepLOB + 2024-2025 transformer successors (HLOB, TLOB) to capture queue toxicity and short-horizon moves.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from collections import deque
import pandas as pd

from ..utils.logger import get_logger


@dataclass
class LOBFeatures:
    """Limit Order Book features."""
    bid_prices: np.ndarray
    ask_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_sizes: np.ndarray
    timestamp: float
    spread: float
    mid_price: float
    order_flow: np.ndarray  # Recent order flow


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure models."""
    sequence_length: int = 100
    num_levels: int = 10
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DeepLOB(nn.Module):
    """
    DeepLOB: Deep Learning for Limit Order Books
    
    Original DeepLOB architecture for microstructure prediction.
    """
    
    def __init__(self, config: MicrostructureConfig):
        super().__init__()
        self.config = config
        
        # Input processing
        self.input_size = config.num_levels * 4  # bid_price, ask_price, bid_size, ask_size
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(32, config.hidden_size, num_layers=config.num_layers, 
                            dropout=config.dropout, batch_first=True)
        self.lstm2 = nn.LSTM(config.hidden_size, config.hidden_size, num_layers=1, 
                            dropout=config.dropout, batch_first=True)
        
        # Output layers
        self.fc1 = nn.Linear(config.hidden_size, 64)
        self.fc2 = nn.Linear(64, 3)  # 3 classes: up, down, stationary
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Reshape for CNN
        x = x.view(batch_size, 1, seq_len, -1)  # (batch, 1, seq_len, features)
        x = x.squeeze(1)  # (batch, seq_len, features)
        
        # Apply convolutions
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Pooling
        x = self.pool(x)
        
        # Reshape for LSTM
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Take last output
        x = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class HLOB(nn.Module):
    """
    HLOB: Hierarchical Limit Order Book Transformer
    
    Hierarchical transformer for capturing multi-scale microstructure patterns.
    """
    
    def __init__(self, config: MicrostructureConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_size = config.num_levels * 4
        self.embedding = nn.Linear(self.input_size, config.hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, config.sequence_length, config.hidden_size))
        
        # Hierarchical transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Multi-scale attention
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(3)  # 3 different scales
        ])
        
        # Output layers
        self.fc1 = nn.Linear(config.hidden_size, 64)
        self.fc2 = nn.Linear(64, 3)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Hierarchical transformer processing
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            
            # Multi-scale attention at different layers
            if i % 2 == 0:  # Apply multi-scale attention every other layer
                multi_scale_out = []
                for attention in self.multi_scale_attention:
                    attn_out, _ = attention(x, x, x)
                    multi_scale_out.append(attn_out)
                
                # Combine multi-scale outputs
                x = torch.stack(multi_scale_out).mean(dim=0)
                x = self.layer_norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TLOB(nn.Module):
    """
    TLOB: Temporal Limit Order Book Transformer
    
    Temporal transformer for capturing time-varying microstructure patterns.
    """
    
    def __init__(self, config: MicrostructureConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_size = config.num_levels * 4
        self.embedding = nn.Linear(self.input_size, config.hidden_size)
        
        # Temporal encoding
        self.temporal_encoding = nn.Parameter(torch.randn(1, config.sequence_length, config.hidden_size))
        
        # Temporal transformer layers
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Temporal attention mechanisms
        self.short_term_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.long_term_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(config.hidden_size, 64)
        self.fc2 = nn.Linear(64, 3)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        x = self.embedding(x)
        
        # Add temporal encoding
        x = x + self.temporal_encoding[:, :seq_len, :]
        
        # Temporal transformer processing
        for i, layer in enumerate(self.temporal_layers):
            x = layer(x)
            
            # Apply temporal attention mechanisms
            if i % 2 == 0:
                # Short-term attention (local patterns)
                short_term_out, _ = self.short_term_attention(x, x, x)
                
                # Long-term attention (global patterns)
                long_term_out, _ = self.long_term_attention(x, x, x)
                
                # Combine temporal patterns
                x = short_term_out + long_term_out
                x = self.layer_norm(x)
        
        # Temporal pooling
        x = x.mean(dim=1)
        
        # Output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MicrostructureEdge:
    """
    Microstructure Edge System
    
    Combines DeepLOB, HLOB, and TLOB for comprehensive microstructure analysis.
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self.logger = get_logger("MicrostructureEdge")
        
        # Initialize models
        self.deeplob = DeepLOB(config).to(config.device)
        self.hlob = HLOB(config).to(config.device)
        self.tlob = TLOB(config).to(config.device)
        
        # Optimizers
        self.optimizer_deeplob = torch.optim.Adam(self.deeplob.parameters(), lr=config.learning_rate)
        self.optimizer_hlob = torch.optim.Adam(self.hlob.parameters(), lr=config.learning_rate)
        self.optimizer_tlob = torch.optim.Adam(self.tlob.parameters(), lr=config.learning_rate)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Data buffers
        self.lob_buffer = deque(maxlen=config.sequence_length)
        self.order_flow_buffer = deque(maxlen=config.sequence_length)
        
        # Model states
        self.models_trained = False
        
    def preprocess_lob_data(self, lob_data: LOBFeatures) -> np.ndarray:
        """Preprocess LOB data for model input."""
        # Combine bid/ask prices and sizes
        features = []
        
        for i in range(self.config.num_levels):
            if i < len(lob_data.bid_prices):
                features.extend([
                    lob_data.bid_prices[i],
                    lob_data.ask_prices[i],
                    lob_data.bid_sizes[i],
                    lob_data.ask_sizes[i]
                ])
            else:
                # Pad with zeros if not enough levels
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def extract_toxicity_features(self, order_flow: np.ndarray) -> Dict[str, float]:
        """Extract queue toxicity features from order flow."""
        if len(order_flow) == 0:
            return {"toxicity": 0.0, "imbalance": 0.0, "pressure": 0.0}
        
        # Order flow imbalance
        buy_volume = np.sum(order_flow[order_flow > 0])
        sell_volume = np.abs(np.sum(order_flow[order_flow < 0]))
        total_volume = buy_volume + sell_volume
        
        imbalance = (buy_volume - sell_volume) / (total_volume + 1e-8)
        
        # Queue toxicity (simplified)
        toxicity = np.std(order_flow) / (np.mean(np.abs(order_flow)) + 1e-8)
        
        # Market pressure
        pressure = np.sum(order_flow) / (len(order_flow) + 1e-8)
        
        return {
            "toxicity": float(toxicity),
            "imbalance": float(imbalance),
            "pressure": float(pressure)
        }
    
    def update_buffers(self, lob_data: LOBFeatures):
        """Update LOB and order flow buffers."""
        features = self.preprocess_lob_data(lob_data)
        self.lob_buffer.append(features)
        self.order_flow_buffer.append(lob_data.order_flow)
    
    def get_model_inputs(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Get inputs for all models."""
        if len(self.lob_buffer) < self.config.sequence_length:
            # Pad with zeros if not enough data
            padding = [np.zeros_like(self.lob_buffer[0]) for _ in range(self.config.sequence_length - len(self.lob_buffer))]
            sequence = list(self.lob_buffer) + padding
        else:
            sequence = list(self.lob_buffer)
        
        # Convert to tensor
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.config.device)
        
        # Extract toxicity features
        recent_order_flow = list(self.order_flow_buffer)[-10:]  # Last 10 order flows
        toxicity_features = self.extract_toxicity_features(np.concatenate(recent_order_flow))
        
        return x, toxicity_features
    
    def predict_microstructure(self, lob_data: LOBFeatures) -> Dict[str, Any]:
        """
        Predict microstructure patterns using ensemble of models.
        
        Args:
            lob_data: Current LOB data
            
        Returns:
            predictions: Ensemble predictions and toxicity analysis
        """
        # Update buffers
        self.update_buffers(lob_data)
        
        # Get model inputs
        x, toxicity_features = self.get_model_inputs()
        
        predictions = {
            "deeplob": None,
            "hlob": None,
            "tlob": None,
            "ensemble": None,
            "toxicity": toxicity_features,
            "confidence": 0.0
        }
        
        if not self.models_trained:
            self.logger.warning("Models not trained yet. Returning default predictions.")
            return predictions
        
        try:
            # Get predictions from all models
            with torch.no_grad():
                deeplob_out = F.softmax(self.deeplob(x), dim=1)
                hlob_out = F.softmax(self.hlob(x), dim=1)
                tlob_out = F.softmax(self.tlob(x), dim=1)
            
            # Store individual predictions
            predictions["deeplob"] = deeplob_out.cpu().numpy()[0]
            predictions["hlob"] = hlob_out.cpu().numpy()[0]
            predictions["tlob"] = tlob_out.cpu().numpy()[0]
            
            # Ensemble prediction (simple average)
            ensemble_pred = (deeplob_out + hlob_out + tlob_out) / 3
            predictions["ensemble"] = ensemble_pred.cpu().numpy()[0]
            
            # Confidence based on agreement
            predictions["confidence"] = float(1.0 - torch.std(torch.stack([deeplob_out, hlob_out, tlob_out])).item())
            
            self.logger.info(f"Microstructure prediction: {predictions['ensemble']}, Confidence: {predictions['confidence']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in microstructure prediction: {str(e)}")
        
        return predictions
    
    def train_models(self, training_data: List[Tuple[LOBFeatures, int]]):
        """
        Train all microstructure models.
        
        Args:
            training_data: List of (LOB_features, label) pairs
        """
        self.logger.info("Starting microstructure model training...")
        
        # Prepare training data
        X, y = [], []
        for lob_data, label in training_data:
            features = self.preprocess_lob_data(lob_data)
            X.append(features)
            y.append(label)
        
        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32).to(self.config.device)
        y = torch.tensor(y, dtype=torch.long).to(self.config.device)
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            # Train DeepLOB
            self.optimizer_deeplob.zero_grad()
            deeplob_out = self.deeplob(X)
            deeplob_loss = self.criterion(deeplob_out, y)
            deeplob_loss.backward()
            self.optimizer_deeplob.step()
            
            # Train HLOB
            self.optimizer_hlob.zero_grad()
            hlob_out = self.hlob(X)
            hlob_loss = self.criterion(hlob_out, y)
            hlob_loss.backward()
            self.optimizer_hlob.step()
            
            # Train TLOB
            self.optimizer_tlob.zero_grad()
            tlob_out = self.tlob(X)
            tlob_loss = self.criterion(tlob_out, y)
            tlob_loss.backward()
            self.optimizer_tlob.step()
            
            if epoch % 2 == 0:
                self.logger.info(f"Epoch {epoch}: DeepLOB={deeplob_loss:.4f}, HLOB={hlob_loss:.4f}, TLOB={tlob_loss:.4f}")
        
        self.models_trained = True
        self.logger.info("Microstructure model training completed.")
    
    def get_short_horizon_signals(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate short-horizon trading signals from microstructure predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            signals: Short-horizon trading signals
        """
        signals = {
            "direction": 0.0,  # -1: short, 0: neutral, 1: long
            "strength": 0.0,   # Signal strength [0, 1]
            "toxicity_adjustment": 0.0
        }
        
        if predictions["ensemble"] is not None:
            ensemble_pred = predictions["ensemble"]
            
            # Direction based on prediction probabilities
            up_prob = ensemble_pred[0]
            down_prob = ensemble_pred[1]
            stationary_prob = ensemble_pred[2]
            
            # Determine direction
            if up_prob > down_prob and up_prob > stationary_prob:
                signals["direction"] = 1.0
                signals["strength"] = up_prob
            elif down_prob > up_prob and down_prob > stationary_prob:
                signals["direction"] = -1.0
                signals["strength"] = down_prob
            else:
                signals["direction"] = 0.0
                signals["strength"] = stationary_prob
            
            # Toxicity adjustment
            toxicity = predictions["toxicity"]["toxicity"]
            signals["toxicity_adjustment"] = max(0.0, 1.0 - toxicity)
            
            # Adjust strength based on toxicity
            signals["strength"] *= signals["toxicity_adjustment"]
        
        return signals