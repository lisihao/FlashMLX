"""
FlashMLX integration layer — protocols and adapters for external consumers.

Provides:
  - FlashMLXProvider: Protocol interface for loose coupling
  - setup_flashmlx(): One-call integration entry point for ThunderOMLX
"""

from .protocol import FlashMLXProvider
from .thunderomlx import setup_flashmlx, flashmlx_settings_schema, ThunderOMLXAdapter

__all__ = [
    "FlashMLXProvider",
    "ThunderOMLXAdapter",
    "setup_flashmlx",
    "flashmlx_settings_schema",
]
