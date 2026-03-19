"""
FlashMLX - High-performance MLX inference engine with Flash Attention optimization
"""

__version__ = "0.1.0"

from . import core
from . import kernels
from . import utils

__all__ = ["core", "kernels", "utils", "__version__"]
