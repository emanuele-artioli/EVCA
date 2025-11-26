"""Numpy-based API for Enhanced Video Complexity Analyzer (EVCA)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch_dct as dct

from .libs.feature_extraction import feature_extraction, temporal_feature_extraction
from .libs.weight_dct import weight_dct


@dataclass
class EVCAConfig:
    """Configuration for EVCA analysis."""
    method: str = "EVCA"  # "EVCA", "VCA"
    transform: str = "DCT"  # "DCT", "DWT", "DCT_B"
    block_size: int = 32
    sample_rate: int = 1
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EVCAResult:
    """Result of EVCA analysis.
    
    All arrays have shape (N, num_blocks_h, num_blocks_w) where:
    - N is the number of frames (after sampling)
    - num_blocks_h = H // block_size
    - num_blocks_w = W // block_size
    """
    B: np.ndarray  # Brightness per block (N, num_blocks_h, num_blocks_w)
    SC: np.ndarray  # Spatial complexity per block
    TC: np.ndarray  # Temporal complexity per block
    TC2: np.ndarray  # Temporal complexity 2 per block
    block_size: int  # Block size used for analysis


class _MockArgs:
    """Mock args object for compatibility with existing functions."""
    def __init__(self, width: int, height: int, config: EVCAConfig):
        self.resolution = f"{width}x{height}"
        self.method = config.method
        self.transform = config.transform
        self.block_size = config.block_size
        self.sample_rate = config.sample_rate


def _frames_to_blocks(frames: torch.Tensor, block_size: int) -> torch.Tensor:
    """Convert frames to blocks for DCT processing.
    
    Args:
        frames: (N, H, W) tensor of Y channel values
        block_size: Size of blocks for DCT
        
    Returns:
        Tensor of shape (N * num_blocks, block_size, block_size)
    """
    N, H, W = frames.shape
    # Ensure dimensions are divisible by block_size
    H_crop = (H // block_size) * block_size
    W_crop = (W // block_size) * block_size
    frames = frames[:, :H_crop, :W_crop]
    
    # Unfold into blocks
    blocks = frames.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    # Shape: (N, H//bs, W//bs, bs, bs)
    blocks = blocks.contiguous().view(-1, block_size, block_size)
    return blocks


def analyze_frames(
    frames: np.ndarray,
    config: Optional[EVCAConfig] = None,
) -> EVCAResult:
    """Analyze video complexity from numpy frames.
    
    Args:
        frames: Video frames as numpy array of shape (N, H, W, 3) or (N, H, W).
                If RGB, will be converted to Y channel. Values should be 0-255, uint8.
        config: EVCA configuration. Uses defaults if None.
        
    Returns:
        EVCAResult with complexity features per block.
        All feature arrays have shape (N, num_blocks_h, num_blocks_w).
        
    Example:
        >>> from evca import analyze_frames, EVCAConfig
        >>> import numpy as np
        >>> 
        >>> frames = np.random.randint(0, 255, (30, 480, 640, 3), dtype=np.uint8)
        >>> result = analyze_frames(frames)
        >>> print(f"SC shape: {result.SC.shape}")  # (30, 15, 20) for block_size=32
    """
    if config is None:
        config = EVCAConfig()
    
    device = torch.device(config.device)
    
    # Handle RGB to Y conversion
    if frames.ndim == 4 and frames.shape[-1] == 3:
        # RGB to Y: Y = 0.299*R + 0.587*G + 0.114*B
        Y = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]).astype(np.uint8)
    elif frames.ndim == 3:
        Y = frames
    else:
        raise ValueError(f"Expected frames of shape (N, H, W, 3) or (N, H, W), got {frames.shape}")
    
    N, H, W = Y.shape
    block_size = config.block_size
    
    # Crop to block-aligned dimensions
    H_crop = (H // block_size) * block_size
    W_crop = (W // block_size) * block_size
    Y = Y[:, :H_crop, :W_crop]
    
    # Create mock args for compatibility
    args = _MockArgs(W_crop, H_crop, config)
    
    # Sample frames according to sample_rate
    frame_indices = np.arange(0, N, config.sample_rate)
    Y_sampled = Y[frame_indices]
    nframes = len(frame_indices)
    
    # Convert to torch
    Y_tensor = torch.from_numpy(Y_sampled).to(device).float()
    
    # Process all frames at once
    blocks = _frames_to_blocks(Y_tensor, block_size)
    
    # Apply DCT
    if config.transform == "DWT":
        from pytorch_wavelets import DWTForward
        dwt = DWTForward().to(device)
        yl, yh = dwt(blocks.unsqueeze(1).float())
        yh = yh[0]
        top_row = torch.cat((yl, yh[:, :, 0, :, :]), dim=3)
        bottom_row = torch.cat((yh[:, :, 1, :, :], yh[:, :, 2, :, :]), dim=3)
        DTs = torch.cat((top_row, bottom_row), dim=2)
    elif config.transform == "DCT_B":
        from .libs import dct_butterfly_torch as dct_b
        DTs = dct_b.dct_32_2d(blocks.type(torch.int32))
    else:
        DTs = dct.dct_2d(blocks)
    
    # Extract features
    B_blocks, SC_blocks, energy = feature_extraction(args, DTs, nframes, device)
    
    # Initialize temporal features storage
    last_energy = torch.tensor([], device=device)
    last_SC = torch.tensor([], device=device)
    
    # Compute temporal features
    TC_blocks, TC2_blocks = temporal_feature_extraction(args, 0, SC_blocks, energy, last_SC, last_energy)
    
    # Reshape to (N, num_blocks_h, num_blocks_w)
    num_blocks_h = H_crop // block_size
    num_blocks_w = W_crop // block_size
    
    B_out = B_blocks.cpu().numpy().reshape(nframes, num_blocks_h, num_blocks_w)
    SC_out = SC_blocks.cpu().numpy().reshape(nframes, num_blocks_h, num_blocks_w)
    
    # TC has one fewer frame (no temporal diff for first frame)
    TC_np = TC_blocks.cpu().numpy().reshape(-1, num_blocks_h, num_blocks_w)
    TC2_np = TC2_blocks.cpu().numpy().reshape(-1, num_blocks_h, num_blocks_w)
    
    # Pad TC with zeros for first frame
    TC_out = np.zeros((nframes, num_blocks_h, num_blocks_w), dtype=TC_np.dtype)
    TC2_out = np.zeros((nframes, num_blocks_h, num_blocks_w), dtype=TC2_np.dtype)
    
    if TC_np.shape[0] > 0:
        TC_out[1:1+TC_np.shape[0]] = TC_np
    if TC2_np.shape[0] > 0:
        TC2_out[2:2+TC2_np.shape[0]] = TC2_np
    
    return EVCAResult(
        B=B_out,
        SC=SC_out,
        TC=TC_out,
        TC2=TC2_out,
        block_size=block_size,
    )
