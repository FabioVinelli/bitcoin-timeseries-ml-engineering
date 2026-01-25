"""
LSTM Model Architecture
=======================
Implements multiple LSTM variants for time-series forecasting:
- Vanilla LSTM
- Bidirectional LSTM
- LSTM with Attention (Lou et al., 2023)
- CNN-Bi-LSTM-Attention (Yang et al., WTED 2023)

References:
    Lou, J., Cui, L., & Li, Y. (2023). Bi-LSTM Price Prediction based on 
    Attention Mechanism. arXiv:2212.03443v2
    
    Yang, Q., Sun, Y., & Wu, Y. (2023). Bitcoin Price Prediction Based on 
    CNN-Bi-LSTM-Attention Model. WTED 2023, Volume 16, pp. 80-86.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """
    Attention mechanism for LSTM outputs.
    
    Based on Lou et al. (2023) Equation 12 and Yang et al. (2023) Section 2.2.3.
    Learns to weight different time steps based on their importance
    for the prediction task.
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 64):
        """
        Parameters
        ----------
        hidden_size : int
            Size of LSTM hidden state
        attention_size : int
            Size of attention layer
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs.
        
        Implements equations from Yang et al. (2023):
            a_t = exp[h_t * h_s] / sum(exp[s(h_t * h_s)])  (Eq. 1)
            c = sum(a_i * h_i)  (Eq. 3)
        
        Parameters
        ----------
        lstm_output : torch.Tensor
            Shape: (batch_size, seq_len, hidden_size)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            context: weighted sum of lstm_output, shape (batch_size, hidden_size)
            attention_weights: shape (batch_size, seq_len)
        """
        # Calculate attention scores
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        
        # Weighted sum of LSTM outputs
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden)
        context = context.squeeze(1)  # (batch, hidden)
        
        return context, attention_weights


class LSTMForecaster(nn.Module):
    """
    LSTM-based model for time-series forecasting (Lou et al., 2023).
    
    Supports:
    - Vanilla LSTM
    - Bidirectional LSTM
    - Attention mechanism
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        attention_size: int = 64,
        fc_layers: list = [256, 64],
        output_size: int = 1
    ):
        """
        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int
            LSTM hidden state size
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        use_attention : bool
            Whether to use attention mechanism
        attention_size : int
            Size of attention layer
        fc_layers : list
            Sizes of fully connected layers
        output_size : int
            Size of output (forecast horizon)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # Input normalization
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention layer
        lstm_output_size = hidden_size * self.num_directions
        if use_attention:
            self.attention = Attention(lstm_output_size, attention_size)
        
        # Fully connected layers
        fc_input_size = lstm_output_size
        layers = []
        for fc_size in fc_layers:
            layers.extend([
                nn.Linear(fc_input_size, fc_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            fc_input_size = fc_size
        layers.append(nn.Linear(fc_input_size, output_size))
        self.fc = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Created LSTMForecaster: input={input_size}, hidden={hidden_size}, "
            f"layers={num_layers}, bidirectional={bidirectional}, attention={use_attention}"
        )
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, seq_len, input_size)
        return_attention : bool
            Whether to return attention weights
            
        Returns
        -------
        torch.Tensor
            Predictions, shape (batch_size, output_size)
            If return_attention=True, also returns attention weights
        """
        batch_size, seq_len, n_features = x.shape
        
        # Apply batch normalization (need to reshape)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden * num_directions)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use last hidden state from both directions
            if self.bidirectional:
                # Concatenate last states from forward and backward
                hidden_forward = hidden[-2, :, :]  # Last forward layer
                hidden_backward = hidden[-1, :, :]  # Last backward layer
                context = torch.cat([hidden_forward, hidden_backward], dim=1)
            else:
                context = hidden[-1, :, :]
            attention_weights = None
        
        # Fully connected layers
        output = self.fc(context)
        
        if return_attention and self.use_attention:
            return output, attention_weights
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (inference mode).
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        torch.Tensor
            Predictions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class CNNBiLSTMAttention(nn.Module):
    """
    CNN-Bi-LSTM-Attention model for time-series forecasting.
    
    Based on Yang et al. (WTED 2023) - achieved R² = 0.991 on Bitcoin prediction.
    
    Architecture (from paper Figure 4):
        Input → Conv1D(64, kernel=3) → MaxPool1D(1) → 
        Bi-LSTM(32) → Bi-LSTM(64) → Attention → Dense(128) → Output
    
    Key findings from paper:
        - Sliding window size = 4 is optimal (Table 1)
        - Conv kernel = 3, Pool kernel = 1 (Table 2)
        - Bi-LSTM layers: 32-64 units (Table 3)
        - Batch size = 256 (Table 4)
    """
    
    def __init__(
        self,
        input_size: int,
        cnn_filters: int = 64,
        cnn_kernel_size: int = 3,
        pool_size: int = 1,
        lstm_units: list = [32, 64],
        dropout: float = 0.3,
        attention_size: int = 128,
        fc_size: int = 128,
        output_size: int = 1
    ):
        """
        Parameters
        ----------
        input_size : int
            Number of input features
        cnn_filters : int
            Number of CNN filters (paper: 64)
        cnn_kernel_size : int
            CNN kernel size (paper: 3)
        pool_size : int
            Max pooling size (paper: 1)
        lstm_units : list
            Hidden units for each Bi-LSTM layer (paper: [32, 64])
        dropout : float
            Dropout rate
        attention_size : int
            Attention dense layer size (paper: 128)
        fc_size : int
            Final FC layer size (paper: 128)
        output_size : int
            Output dimension
        """
        super().__init__()
        
        self.input_size = input_size
        
        # CNN for feature extraction (paper Section 2.2.1)
        # Conv1D expects (batch, channels, seq_len), we have (batch, seq_len, features)
        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2  # Same padding
        )
        self.pool = nn.MaxPool1d(kernel_size=pool_size) if pool_size > 1 else nn.Identity()
        self.cnn_bn = nn.BatchNorm1d(cnn_filters)
        
        # Two-layer Bi-LSTM (paper Section 3.2.3, Table 3)
        self.lstm1 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units[0],
            batch_first=True,
            bidirectional=True
        )
        self.lstm_dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=lstm_units[0] * 2,  # Bidirectional doubles output
            hidden_size=lstm_units[1],
            batch_first=True,
            bidirectional=True
        )
        self.lstm_dropout2 = nn.Dropout(dropout)
        
        # Attention mechanism (paper Section 2.2.3)
        lstm_output_size = lstm_units[1] * 2  # Bidirectional
        self.attention_dense = nn.Linear(lstm_output_size, attention_size)
        self.attention_weights = nn.Linear(attention_size, 1)
        
        # Output layers (paper Figure 4: Dense(128) → Dense(1))
        self.fc1 = nn.Linear(lstm_output_size, fc_size)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(fc_size, output_size)
        
        logger.info(
            f"Created CNNBiLSTMAttention: input={input_size}, "
            f"cnn_filters={cnn_filters}, lstm_units={lstm_units}"
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, seq_len, input_size)
        return_attention : bool
            Whether to return attention weights
            
        Returns
        -------
        torch.Tensor
            Predictions, shape (batch_size, output_size)
        """
        batch_size, seq_len, n_features = x.shape
        
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        
        # CNN feature extraction
        x = self.conv1d(x)  # (batch, cnn_filters, seq_len)
        x = F.relu(x)
        x = self.pool(x)
        x = self.cnn_bn(x)
        
        # Back to (batch, seq_len, features) for LSTM
        x = x.permute(0, 2, 1)
        
        # First Bi-LSTM layer
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.lstm_dropout1(lstm_out1)
        
        # Second Bi-LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.lstm_dropout2(lstm_out2)
        
        # Attention mechanism (paper Equations 1-3)
        attention_scores = torch.tanh(self.attention_dense(lstm_out2))
        attention_scores = self.attention_weights(attention_scores).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum (context vector)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out2).squeeze(1)
        
        # Output layers
        out = F.relu(self.fc1(context))
        out = self.fc_dropout(out)
        out = self.fc_out(out)
        
        if return_attention:
            return out, attention_weights
        return out


class SimpleLSTM(nn.Module):
    """
    Simplified LSTM for baseline comparison.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def create_model(config: dict) -> nn.Module:
    """
    Factory function to create model based on config.
    
    Parameters
    ----------
    config : dict
        Model configuration with 'type' key:
        - 'simple': SimpleLSTM baseline
        - 'lstm': Vanilla LSTM
        - 'bilstm': Bidirectional LSTM
        - 'lstm_attention' / 'at_bilstm': At-BiLSTM (Lou et al., 2023)
        - 'cnn_bilstm_attention': CNN-Bi-LSTM-Attention (Yang et al., 2023)
        
    Returns
    -------
    nn.Module
        Created model
    """
    model_type = config.get("type", "lstm_attention").lower()
    
    if model_type == "simple":
        return SimpleLSTM(
            input_size=config["input_size"],
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.2),
            output_size=config.get("output_size", 1)
        )
    
    elif model_type == "cnn_bilstm_attention":
        # Yang et al. (WTED 2023) architecture
        return CNNBiLSTMAttention(
            input_size=config["input_size"],
            cnn_filters=config.get("cnn_filters", 64),
            cnn_kernel_size=config.get("cnn_kernel_size", 3),
            pool_size=config.get("pool_size", 1),
            lstm_units=config.get("lstm_units", [32, 64]),
            dropout=config.get("dropout", 0.3),
            attention_size=config.get("attention_size", 128),
            fc_size=config.get("fc_size", 128),
            output_size=config.get("output_size", 1)
        )
    
    elif model_type in ["lstm", "bilstm", "lstm_attention", "at_bilstm"]:
        # Lou et al. (2023) architecture variants
        return LSTMForecaster(
            input_size=config["input_size"],
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
            bidirectional=config.get("bidirectional", model_type in ["bilstm", "lstm_attention", "at_bilstm"]),
            use_attention=config.get("use_attention", model_type in ["lstm_attention", "at_bilstm"]),
            attention_size=config.get("attention_size", 64),
            fc_layers=config.get("fc_layers", [256, 64]),
            output_size=config.get("output_size", 1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: simple, lstm, bilstm, lstm_attention, at_bilstm, cnn_bilstm_attention")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    config = {
        "type": "lstm_attention",
        "input_size": 30,
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "use_attention": True
    }
    
    model = create_model(config)
    print(model)
    
    # Test forward pass
    batch_size = 16
    seq_len = 60
    x = torch.randn(batch_size, seq_len, config["input_size"])
    
    output, attention = model(x, return_attention=True)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attention.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
