"""
CS2 prediction models
Swin-Tiny backbone + task-specific classification heads
"""
import torch
import torch.nn as nn
import timm


class CS2Predictor(nn.Module):
    """
    CS2 round predictor with Swin-Tiny backbone.
    Single-frame mode.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        img_size: int = 224,
        dropout: float = 0.3,
        model_name: str = "swin_tiny_patch4_window7_224",
    ):
        super().__init__()
        
        # ConvNeXt does not accept img_size; Swin does
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if "swin" in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        
        feat_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
    
    def forward(self, x):
        # x: [B, C, H, W] for single frame
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class CS2SequencePredictor(nn.Module):
    """
    CS2 round predictor with Swin-Tiny + causal temporal modeling.
    
    Causal constraint: the model can only look at past frames,
    never future frames. This is enforced by:
    1. Unidirectional (causal) GRU
    2. Using the last timestep's hidden state for prediction
    
    During inference for online prediction, feed the last X frames
    and the model outputs a prediction based only on those X frames.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        img_size: int = 224,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        model_name: str = "swin_tiny_patch4_window7_224",
    ):
        super().__init__()
        
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if "swin" in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        
        feat_dim = self.backbone.num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Causal (unidirectional) GRU - can only see past
        self.temporal = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # CAUSAL: no peeking into future
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x, valid_mask=None):
        """
        Args:
            x: [B, T, C, H, W] sequence of frames
            valid_mask: [B, T] bool mask, True for valid frames
        Returns:
            logits: [B, num_classes]
        """
        B, T, C, H, W = x.shape
        
        # Extract visual features per frame (shared backbone)
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # [B*T, feat_dim]
        features = features.view(B, T, -1)  # [B, T, feat_dim]
        
        # Causal temporal modeling
        if valid_mask is not None:
            # Pack padded sequences for efficient GRU computation
            lengths = valid_mask.sum(dim=1).cpu().long()
            # Sort by length descending (required by pack_padded_sequence)
            lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
            features_sorted = features[sort_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                features_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            gru_out, hidden = self.temporal(packed)
            # hidden: [num_layers, B, hidden_dim]
            # Take the last layer's final hidden state
            last_hidden = hidden[-1]  # [B, hidden_dim]
            # Unsort back to original order
            _, unsort_idx = torch.sort(sort_idx)
            last_hidden = last_hidden[unsort_idx]
        else:
            gru_out, hidden = self.temporal(features)
            last_hidden = hidden[-1]  # [B, hidden_dim]
        
        logits = self.classifier(last_hidden)
        return logits


class CS2WinProbPredictor(nn.Module):
    """
    Winner prediction with win probability output.
    Same architecture as CS2Predictor but outputs 2 probabilities
    that sum to 1 (via softmax).
    
    The probabilities can be interpreted as T/CT win rates.
    Not directly verifiable against ground truth, but useful for visualization.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        img_size: int = 224,
        dropout: float = 0.3,
        model_name: str = "swin_tiny_patch4_window7_224",
    ):
        super().__init__()
        
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if "swin" in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        
        feat_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2),  # [logit_ct, logit_t]
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)  # [B, 2]
        probs = torch.softmax(logits, dim=-1)  # [B, 2], sum to 1
        return logits, probs
