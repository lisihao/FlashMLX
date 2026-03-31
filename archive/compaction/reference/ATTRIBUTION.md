# Attribution

This directory contains code from the Attention Matching project:

- **Original Repository**: https://github.com/adamzweiger/compaction
- **License**: MIT License
- **Authors**: Adam Zweiger and contributors
- **Paper**: "Fast KV Compaction via Attention Matching" (2026)
  - arXiv: https://arxiv.org/abs/2602.16284

## Modifications

The original PyTorch implementation has been adapted for use with MLX through:
1. A thin adapter layer (`torch_to_mlx_adapter.py`) that translates PyTorch tensors to MLX arrays
2. Minimal changes to accommodate API differences between PyTorch and MLX
3. All algorithmic logic remains unchanged from the original implementation

## License

The original code is licensed under the MIT License. See the LICENSE file in the `compaction/` directory for details.
