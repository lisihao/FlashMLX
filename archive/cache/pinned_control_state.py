"""
Pinned Control State Manager

Identifies and protects control channels (language, format, think mode)
that should never be evicted from memory.
"""

from dataclasses import dataclass
from typing import Set, List, Optional
from enum import Enum
import re


class ControlChannelType(Enum):
    """Types of control channels to protect"""
    LANGUAGE = "language"       # Language selection tokens (e.g., "Chinese", "English")
    FORMAT = "format"          # Format control (e.g., "list", "table", "JSON")
    THINK_MODE = "think_mode"  # Think tag control (e.g., "<think>", "</think>")
    SYSTEM = "system"          # System instructions
    DELIMITER = "delimiter"    # Section delimiters (e.g., "---", "###")


@dataclass
class ControlChannel:
    """
    Represents a control channel that should be pinned in memory.

    Args:
        channel_type: Type of control channel
        position: Token position in sequence
        importance: Importance score (0.0 - 1.0)
        content: Optional content snapshot for debugging
    """
    channel_type: ControlChannelType
    position: int
    importance: float
    content: Optional[str] = None

    def __post_init__(self):
        """Validate control channel"""
        if self.importance < 0.0 or self.importance > 1.0:
            raise ValueError(f"importance must be in [0, 1], got {self.importance}")

        if self.position < 0:
            raise ValueError(f"position must be non-negative, got {self.position}")


class PinnedControlState:
    """
    Manages pinned control channels that should never be evicted.

    Control channels are special tokens/positions that control:
    - Language selection (e.g., "请用中文回答")
    - Output format (e.g., "使用列表格式")
    - Think mode (e.g., "<think>" tags)
    - System instructions

    These channels are identified by pattern matching and importance scoring,
    then protected from eviction in the memory management system.

    Example:
        >>> manager = PinnedControlState()
        >>> # Detect control channels in text
        >>> text = "请用中文回答。<think>分析问题...</think>答案是..."
        >>> channels = manager.detect_control_channels(text)
        >>> len(channels)
        3  # language + think_start + think_end
        >>> manager.is_pinned(0)  # Position 0 (language instruction)
        True
    """

    def __init__(self, max_pinned_positions: int = 100):
        """
        Initialize Pinned Control State manager.

        Args:
            max_pinned_positions: Maximum number of positions to pin
        """
        self.max_pinned_positions = max_pinned_positions

        # Pinned positions (set of token positions)
        self.pinned_positions: Set[int] = set()

        # Control channels (detailed information)
        self.control_channels: List[ControlChannel] = []

        # Pattern definitions for control channel detection
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for control channel detection"""
        self.patterns = {
            ControlChannelType.LANGUAGE: [
                r"(请用|请使用|use|answer in|回答用)\s*(中文|英文|English|Chinese)",
                r"(中文|English)\s*(回答|answer)",
            ],
            ControlChannelType.FORMAT: [
                r"(使用|use|format|格式)\s*(列表|表格|JSON|list|table|bullet)",
                r"(列出|list)\s*\d+",
            ],
            ControlChannelType.THINK_MODE: [
                r"<think>",
                r"</think>",
                r"<thinking>",
                r"</thinking>",
            ],
            ControlChannelType.DELIMITER: [
                r"^---+$",
                r"^===+$",
                r"^###",
            ],
        }

    def detect_control_channels(
        self,
        text: str,
        importance_threshold: float = 0.5
    ) -> List[ControlChannel]:
        """
        Detect control channels in text.

        Args:
            text: Input text to analyze
            importance_threshold: Minimum importance to consider

        Returns:
            List of detected control channels
        """
        detected_channels = []

        # Detect each channel type
        for channel_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)

                for match in matches:
                    position = match.start()
                    content = match.group()

                    # Calculate importance based on channel type and position
                    importance = self._calculate_importance(
                        channel_type=channel_type,
                        position=position,
                        text_length=len(text)
                    )

                    if importance >= importance_threshold:
                        channel = ControlChannel(
                            channel_type=channel_type,
                            position=position,
                            importance=importance,
                            content=content
                        )
                        detected_channels.append(channel)

        return detected_channels

    def _calculate_importance(
        self,
        channel_type: ControlChannelType,
        position: int,
        text_length: int
    ) -> float:
        """
        Calculate importance score for a control channel.

        Importance is higher for:
        - Channels at the beginning of text
        - Critical channel types (language, think mode)

        Args:
            channel_type: Type of control channel
            position: Position in text
            text_length: Total text length

        Returns:
            Importance score (0.0 - 1.0)
        """
        # Base importance by channel type
        type_importance = {
            ControlChannelType.LANGUAGE: 1.0,      # Highest priority
            ControlChannelType.THINK_MODE: 0.9,    # Very important
            ControlChannelType.FORMAT: 0.8,        # Important
            ControlChannelType.SYSTEM: 1.0,        # Critical
            ControlChannelType.DELIMITER: 0.5,     # Medium
        }

        base_importance = type_importance.get(channel_type, 0.5)

        # Position bonus: earlier positions are more important
        # Decay from 1.0 at position 0 to 0.5 at end
        if text_length > 0:
            position_bonus = 1.0 - (0.5 * position / text_length)
        else:
            position_bonus = 1.0

        # Combined importance
        importance = base_importance * position_bonus

        return min(1.0, importance)

    def pin_channels(self, channels: List[ControlChannel]):
        """
        Pin control channels to prevent eviction.

        Args:
            channels: List of control channels to pin
        """
        # Sort by importance
        sorted_channels = sorted(channels, key=lambda c: c.importance, reverse=True)

        # Pin top N channels
        for channel in sorted_channels[:self.max_pinned_positions]:
            self.pinned_positions.add(channel.position)
            self.control_channels.append(channel)

    def is_pinned(self, position: int) -> bool:
        """
        Check if a position is pinned.

        Args:
            position: Token position

        Returns:
            True if position is pinned
        """
        return position in self.pinned_positions

    def unpin(self, position: int) -> bool:
        """
        Unpin a position.

        Args:
            position: Token position

        Returns:
            True if position was pinned and is now unpinned
        """
        if position in self.pinned_positions:
            self.pinned_positions.remove(position)

            # Remove from control channels
            self.control_channels = [
                c for c in self.control_channels if c.position != position
            ]

            return True

        return False

    def get_pinned_count(self) -> int:
        """Get number of pinned positions"""
        return len(self.pinned_positions)

    def get_pinned_positions(self) -> List[int]:
        """Get all pinned positions (sorted)"""
        return sorted(self.pinned_positions)

    def get_channels_by_type(
        self,
        channel_type: ControlChannelType
    ) -> List[ControlChannel]:
        """
        Get all control channels of a specific type.

        Args:
            channel_type: Type of control channel

        Returns:
            List of matching control channels
        """
        return [c for c in self.control_channels if c.channel_type == channel_type]

    def clear(self):
        """Clear all pinned positions and control channels"""
        self.pinned_positions.clear()
        self.control_channels.clear()

    def get_statistics(self) -> dict:
        """
        Get pinned control state statistics.

        Returns:
            Dictionary with statistics
        """
        type_counts = {}
        for channel_type in ControlChannelType:
            type_counts[channel_type.value] = len(
                self.get_channels_by_type(channel_type)
            )

        return {
            "total_pinned": self.get_pinned_count(),
            "max_pinned": self.max_pinned_positions,
            "utilization": self.get_pinned_count() / self.max_pinned_positions,
            "channels_by_type": type_counts,
            "avg_importance": (
                sum(c.importance for c in self.control_channels) / len(self.control_channels)
                if self.control_channels else 0.0
            ),
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"PinnedControlState("
            f"pinned={stats['total_pinned']}/{stats['max_pinned']}, "
            f"avg_importance={stats['avg_importance']:.2f})"
        )
