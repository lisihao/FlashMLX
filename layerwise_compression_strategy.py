#!/usr/bin/env python3
"""
Layerwise Compression Ratio Strategy

Based on user feedback:
"前层可压，后层危险 → 前层3x-5x，中层1.5x-2x，后层1x或1.1x"

Motivation:
- Early layers: Pattern-based, high redundancy → aggressive compression (3x-5x)
- Middle layers: Transition zone → moderate compression (1.5x-2x)
- Late layers: Context-critical, low redundancy → conservative (1x-1.1x)

论文支持: Nonuniform budget allocation is more effective than uniform.
"""

import numpy as np
from typing import List, Tuple


def get_layerwise_ratios_linear(
    num_layers: int,
    early_ratio: float = 5.0,
    late_ratio: float = 1.1
) -> List[float]:
    """
    Linear interpolation from early to late layers.

    Parameters
    ----------
    num_layers : int
        Total number of layers
    early_ratio : float
        Compression ratio for first layer (default: 5.0)
    late_ratio : float
        Compression ratio for last layer (default: 1.1)

    Returns
    -------
    ratios : List[float]
        Compression ratio for each layer

    Example
    -------
    >>> ratios = get_layerwise_ratios_linear(36, early_ratio=5.0, late_ratio=1.1)
    >>> # Layer 0: 5.0, Layer 35: 1.1, Linear interpolation in between
    """
    return list(np.linspace(early_ratio, late_ratio, num_layers))


def get_layerwise_ratios_stepped(
    num_layers: int,
    early_ratio: float = 5.0,
    mid_ratio: float = 2.0,
    late_ratio: float = 1.1
) -> List[float]:
    """
    Three-stage stepped strategy: Early (aggressive) → Mid (moderate) → Late (conservative).

    Parameters
    ----------
    num_layers : int
        Total number of layers
    early_ratio : float
        Compression ratio for early layers (default: 5.0)
    mid_ratio : float
        Compression ratio for middle layers (default: 2.0)
    late_ratio : float
        Compression ratio for late layers (default: 1.1)

    Returns
    -------
    ratios : List[float]
        Compression ratio for each layer

    Strategy
    --------
    - First 1/3: early_ratio (5.0x)
    - Middle 1/3: mid_ratio (2.0x)
    - Last 1/3: late_ratio (1.1x)

    Example
    -------
    >>> ratios = get_layerwise_ratios_stepped(36)
    >>> # Layers 0-11: 5.0x, Layers 12-23: 2.0x, Layers 24-35: 1.1x
    """
    ratios = []
    early_end = num_layers // 3
    mid_end = 2 * num_layers // 3

    for i in range(num_layers):
        if i < early_end:
            ratios.append(early_ratio)
        elif i < mid_end:
            ratios.append(mid_ratio)
        else:
            ratios.append(late_ratio)

    return ratios


def get_layerwise_ratios_exponential(
    num_layers: int,
    early_ratio: float = 5.0,
    late_ratio: float = 1.1
) -> List[float]:
    """
    Exponential decay from early to late layers.

    Parameters
    ----------
    num_layers : int
        Total number of layers
    early_ratio : float
        Compression ratio for first layer (default: 5.0)
    late_ratio : float
        Compression ratio for last layer (default: 1.1)

    Returns
    -------
    ratios : List[float]
        Compression ratio for each layer

    Strategy
    --------
    Exponential decay provides smoother transition than linear or stepped.
    More aggressive compression in early layers, tapering off gradually.

    Example
    -------
    >>> ratios = get_layerwise_ratios_exponential(36)
    >>> # Layer 0: 5.0, Layer 35: 1.1, Exponential decay in between
    """
    # log(ratio) = a * layer_idx + b
    # Solve for a, b using boundary conditions
    log_early = np.log(early_ratio)
    log_late = np.log(late_ratio)

    a = (log_late - log_early) / (num_layers - 1)
    b = log_early

    ratios = []
    for i in range(num_layers):
        log_ratio = a * i + b
        ratios.append(np.exp(log_ratio))

    return ratios


def get_layerwise_ratios_custom(
    num_layers: int,
    strategy: str = "stepped"
) -> Tuple[List[float], str]:
    """
    Get layerwise compression ratios using specified strategy.

    Parameters
    ----------
    num_layers : int
        Total number of layers
    strategy : str
        Strategy name: "linear", "stepped", "exponential", "uniform"

    Returns
    -------
    ratios : List[float]
        Compression ratio for each layer
    description : str
        Strategy description

    Example
    -------
    >>> ratios, desc = get_layerwise_ratios_custom(36, strategy="stepped")
    >>> print(desc)
    Stepped: Early 5.0x, Mid 2.0x, Late 1.1x
    """
    if strategy == "linear":
        ratios = get_layerwise_ratios_linear(num_layers, early_ratio=5.0, late_ratio=1.1)
        description = "Linear: 5.0x → 1.1x"
    elif strategy == "stepped":
        ratios = get_layerwise_ratios_stepped(num_layers, early_ratio=5.0, mid_ratio=2.0, late_ratio=1.1)
        description = "Stepped: Early 5.0x, Mid 2.0x, Late 1.1x"
    elif strategy == "exponential":
        ratios = get_layerwise_ratios_exponential(num_layers, early_ratio=5.0, late_ratio=1.1)
        description = "Exponential: 5.0x → 1.1x"
    elif strategy == "uniform":
        ratios = [1.5] * num_layers
        description = "Uniform: 1.5x (baseline)"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return ratios, description


def visualize_layerwise_ratios(num_layers: int = 36):
    """
    Visualize all layerwise compression strategies.

    Example
    -------
    >>> visualize_layerwise_ratios(36)
    """
    strategies = ["uniform", "linear", "stepped", "exponential"]

    print("=" * 80)
    print("Layerwise Compression Ratio Strategies")
    print("=" * 80)
    print(f"Number of layers: {num_layers}\n")

    for strategy in strategies:
        ratios, description = get_layerwise_ratios_custom(num_layers, strategy=strategy)

        print(f"Strategy: {description}")
        print(f"  Layer 0-11 (Early):  {ratios[0]:.2f}x - {ratios[11]:.2f}x")
        print(f"  Layer 12-23 (Mid):   {ratios[12]:.2f}x - {ratios[23]:.2f}x")
        print(f"  Layer 24-35 (Late):  {ratios[24]:.2f}x - {ratios[35]:.2f}x")
        print(f"  Average:             {np.mean(ratios):.2f}x")
        print()

    print("=" * 80)


if __name__ == "__main__":
    # Visualize strategies
    visualize_layerwise_ratios(36)

    # Example usage
    print("\nExample: Get stepped ratios for 36-layer model")
    ratios, desc = get_layerwise_ratios_custom(36, strategy="stepped")
    print(f"Strategy: {desc}")
    print(f"Ratios: {ratios[:5]} ... {ratios[-5:]}")
