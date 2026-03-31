#!/usr/bin/env python3
"""Debug beta compensation."""

import sys
sys.path.insert(0, '/Users/lisihao/FlashMLX/mlx-lm-source')

import mlx.core as mx
import pickle

# Load calibration file
calib_file = "calibrations/am_calibration_qwen3-8b_2.0x_onpolicy.pkl"

with open(calib_file, 'rb') as f:
    calib_data = pickle.load(f)

# Handle format
if isinstance(calib_data, dict) and 'calibration' in calib_data:
    calibration = calib_data['calibration']
else:
    calibration = calib_data

# Check layer 0
layer0 = calibration[0]

print("Layer 0 Calibration:")
print(f"  Budget: {layer0['budget']}")
print(f"  Compression ratio: {layer0['compression_ratio']}")
print(f"  Beta shape: {layer0['beta'].shape}")
print(f"  Beta min/max: {layer0['beta'].min():.4f} / {layer0['beta'].max():.4f}")
print(f"  Beta mean: {layer0['beta'].mean():.4f}")
print(f"  Selected indices shape: {layer0['selected_indices'].shape}")
print(f"  Ck shape: {layer0['Ck'].shape}")

# Check if beta values are reasonable
import numpy as np
beta_np = np.array(layer0['beta'])
print(f"\nBeta statistics:")
print(f"  < 0.5: {(beta_np < 0.5).sum()}")
print(f"  0.5-1.0: {((beta_np >= 0.5) & (beta_np < 1.0)).sum()}")
print(f"  1.0-1.5: {((beta_np >= 1.0) & (beta_np < 1.5)).sum()}")
print(f"  > 1.5: {(beta_np >= 1.5).sum()}")
