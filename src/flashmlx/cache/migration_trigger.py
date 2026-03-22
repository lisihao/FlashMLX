"""
Migration Trigger

⚠️  DEPRECATED: 2026-03-22 - SSM cache sealed. See SSM_CACHE_DEPRECATION.md


Determines when to migrate cache entries between Hot/Warm/Cold tiers.
Uses semantic boundaries, chunk prediction, and waterline monitoring.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import re


class MigrationType(Enum):
    """Type of migration operation"""
    HOT_TO_WARM = "hot_to_warm"      # Demotion from Hot to Warm
    WARM_TO_COLD = "warm_to_cold"    # Demotion from Warm to Cold
    COLD_TO_WARM = "cold_to_warm"    # Revival from Cold to Warm
    WARM_TO_HOT = "warm_to_hot"      # Promotion from Warm to Hot


@dataclass
class MigrationDecision:
    """
    Migration decision result.

    Args:
        migration_type: Type of migration
        layer_indices: List of layer indices to migrate
        reason: Human-readable reason for migration
        urgency: Urgency score (0.0-1.0), higher = more urgent
    """
    migration_type: MigrationType
    layer_indices: List[int]
    reason: str
    urgency: float

    def __post_init__(self):
        """Validate urgency range"""
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError(f"urgency must be in [0.0, 1.0], got {self.urgency}")


class SemanticBoundaryDetector:
    """
    Detects semantic boundaries in token sequences.

    Semantic boundaries are points where it's safe to migrate cache entries
    without breaking coherence.

    Example:
        >>> detector = SemanticBoundaryDetector()
        >>> tokens = ["Hello", "world", ".", "\\n", "New", "paragraph"]
        >>> detector.is_boundary(tokens, 3)  # After "."
        True
    """

    def __init__(self):
        """Initialize semantic boundary detector"""
        # Sentence-ending punctuation
        self.sentence_enders = {'.', '!', '?', '。', '！', '？'}

        # Paragraph markers
        self.paragraph_markers = {'\n\n', '\n', '<|im_end|>', '</s>'}

        # Token patterns that indicate boundaries
        self.boundary_patterns = [
            r'^[.!?。！？]+$',           # Punctuation only
            r'^\n+$',                     # Newlines only
            r'^<\|im_end\|>$',           # Chat template end
            r'^</s>$',                    # EOS token
        ]

    def is_boundary(self, tokens: List[str], position: int) -> bool:
        """
        Check if position is at a semantic boundary.

        Args:
            tokens: List of token strings
            position: Position to check (0-indexed)

        Returns:
            True if position is at a boundary
        """
        if position < 0 or position >= len(tokens):
            return False

        token = tokens[position]

        # Check sentence enders
        if token in self.sentence_enders:
            return True

        # Check paragraph markers
        if token in self.paragraph_markers:
            return True

        # Check patterns
        for pattern in self.boundary_patterns:
            if re.match(pattern, token):
                return True

        return False

    def find_next_boundary(self, tokens: List[str], start: int = 0) -> Optional[int]:
        """
        Find next semantic boundary after start position.

        Args:
            tokens: List of token strings
            start: Starting position (inclusive)

        Returns:
            Index of next boundary, or None if not found
        """
        for i in range(start, len(tokens)):
            if self.is_boundary(tokens, i):
                return i
        return None

    def get_boundaries(self, tokens: List[str]) -> List[int]:
        """
        Get all semantic boundaries in token sequence.

        Args:
            tokens: List of token strings

        Returns:
            List of boundary indices
        """
        boundaries = []
        for i, token in enumerate(tokens):
            if self.is_boundary(tokens, i):
                boundaries.append(i)
        return boundaries


class ChunkPredictor:
    """
    Predicts whether next token belongs to a new semantic chunk.

    Helps determine if it's safe to migrate cache at current position.

    Example:
        >>> predictor = ChunkPredictor()
        >>> predictor.is_new_chunk(last_token=".", current_token="The")
        True
    """

    def __init__(self):
        """Initialize chunk predictor"""
        self.boundary_detector = SemanticBoundaryDetector()

    def is_new_chunk(self, last_token: str, current_token: str) -> bool:
        """
        Predict if current token starts a new chunk.

        Args:
            last_token: Previous token
            current_token: Current token

        Returns:
            True if current token likely starts new chunk
        """
        # Boundary token followed by anything = new chunk
        if last_token in self.boundary_detector.sentence_enders:
            return True

        if last_token in self.boundary_detector.paragraph_markers:
            return True

        # Newline followed by non-whitespace = new chunk
        if last_token == '\n' and current_token.strip():
            return True

        # EOS token followed by anything = new chunk
        if last_token in {'<|im_end|>', '</s>'}:
            return True

        return False


class WaterlineMonitor:
    """
    Monitors tier utilization and determines migration urgency.

    Waterlines are utilization thresholds that trigger migrations:
    - High waterline (>80%): Trigger demotion to free space
    - Low waterline (<30%): Consider promotion if there are good candidates

    Example:
        >>> monitor = WaterlineMonitor()
        >>> monitor.check_hot_tier(utilization=0.85, entry_count=100)
        MigrationDecision(HOT_TO_WARM, [...], "Hot tier over high waterline", 0.85)
    """

    def __init__(
        self,
        hot_high_waterline: float = 0.80,
        warm_high_waterline: float = 0.80,
        warm_low_waterline: float = 0.30
    ):
        """
        Initialize waterline monitor.

        Args:
            hot_high_waterline: Hot tier demotion threshold
            warm_high_waterline: Warm tier demotion threshold
            warm_low_waterline: Warm tier promotion threshold
        """
        if not 0.0 < hot_high_waterline <= 1.0:
            raise ValueError(f"hot_high_waterline must be in (0, 1], got {hot_high_waterline}")
        if not 0.0 < warm_high_waterline <= 1.0:
            raise ValueError(f"warm_high_waterline must be in (0, 1], got {warm_high_waterline}")
        if not 0.0 <= warm_low_waterline < 1.0:
            raise ValueError(f"warm_low_waterline must be in [0, 1), got {warm_low_waterline}")

        self.hot_high_waterline = hot_high_waterline
        self.warm_high_waterline = warm_high_waterline
        self.warm_low_waterline = warm_low_waterline

    def check_hot_tier(
        self,
        utilization: float,
        demotion_candidates: List[int]
    ) -> Optional[MigrationDecision]:
        """
        Check Hot tier for demotion needs.

        Args:
            utilization: Current Hot tier utilization (0.0-1.0)
            demotion_candidates: List of candidate layer indices (from LRU)

        Returns:
            MigrationDecision if demotion needed, None otherwise
        """
        if utilization <= self.hot_high_waterline:
            return None

        if not demotion_candidates:
            return None

        # Urgency increases with utilization
        urgency = min(1.0, (utilization - self.hot_high_waterline) / (1.0 - self.hot_high_waterline))

        return MigrationDecision(
            migration_type=MigrationType.HOT_TO_WARM,
            layer_indices=demotion_candidates,
            reason=f"Hot tier utilization {utilization:.1%} exceeds high waterline {self.hot_high_waterline:.1%}",
            urgency=urgency
        )

    def check_warm_tier_demotion(
        self,
        utilization: float,
        demotion_candidates: List[int]
    ) -> Optional[MigrationDecision]:
        """
        Check Warm tier for demotion needs.

        Args:
            utilization: Current Warm tier utilization (0.0-1.0)
            demotion_candidates: List of candidate layer indices (from demotion scores)

        Returns:
            MigrationDecision if demotion needed, None otherwise
        """
        if utilization <= self.warm_high_waterline:
            return None

        if not demotion_candidates:
            return None

        urgency = min(1.0, (utilization - self.warm_high_waterline) / (1.0 - self.warm_high_waterline))

        return MigrationDecision(
            migration_type=MigrationType.WARM_TO_COLD,
            layer_indices=demotion_candidates,
            reason=f"Warm tier utilization {utilization:.1%} exceeds high waterline {self.warm_high_waterline:.1%}",
            urgency=urgency
        )

    def check_warm_tier_promotion(
        self,
        utilization: float,
        promotion_candidates: List[int]
    ) -> Optional[MigrationDecision]:
        """
        Check Warm tier for promotion opportunities.

        Args:
            utilization: Current Warm tier utilization (0.0-1.0)
            promotion_candidates: List of candidate layer indices (from promotion scores)

        Returns:
            MigrationDecision if promotion beneficial, None otherwise
        """
        # Only promote if Warm has space
        if utilization >= self.warm_high_waterline:
            return None

        if not promotion_candidates:
            return None

        # Lower urgency for promotions (not critical)
        urgency = 0.3

        return MigrationDecision(
            migration_type=MigrationType.WARM_TO_HOT,
            layer_indices=promotion_candidates,
            reason=f"Warm tier utilization {utilization:.1%} below threshold, promote hot candidates",
            urgency=urgency
        )

    def check_cold_revival(
        self,
        warm_utilization: float,
        revival_candidates: List[int]
    ) -> Optional[MigrationDecision]:
        """
        Check Cold tier for revival opportunities.

        Args:
            warm_utilization: Current Warm tier utilization (0.0-1.0)
            revival_candidates: List of candidate layer indices (from access count)

        Returns:
            MigrationDecision if revival beneficial, None otherwise
        """
        # Only revive if Warm has space
        if warm_utilization >= self.warm_low_waterline:
            return None

        if not revival_candidates:
            return None

        # Low urgency for revivals
        urgency = 0.2

        return MigrationDecision(
            migration_type=MigrationType.COLD_TO_WARM,
            layer_indices=revival_candidates,
            reason=f"Warm tier has space ({warm_utilization:.1%}), revive accessed entries",
            urgency=urgency
        )


class MigrationTrigger:
    """
    Unified migration trigger that combines all strategies.

    Determines when and what to migrate based on:
    1. Semantic boundaries (safe points to migrate)
    2. Chunk prediction (new chunk = good migration point)
    3. Waterline monitoring (utilization-based triggers)

    Example:
        >>> trigger = MigrationTrigger()
        >>> decisions = trigger.evaluate(
        ...     tokens=["Hello", "world", "."],
        ...     hot_util=0.85,
        ...     warm_util=0.60,
        ...     hot_demotion_candidates=[0, 1],
        ...     warm_promotion_candidates=[5]
        ... )
        >>> for decision in decisions:
        ...     print(f"{decision.migration_type}: {len(decision.layer_indices)} layers")
    """

    def __init__(
        self,
        hot_high_waterline: float = 0.80,
        warm_high_waterline: float = 0.80,
        warm_low_waterline: float = 0.30,
        enable_semantic_gating: bool = True
    ):
        """
        Initialize migration trigger.

        Args:
            hot_high_waterline: Hot tier demotion threshold
            warm_high_waterline: Warm tier demotion threshold
            warm_low_waterline: Warm tier promotion threshold
            enable_semantic_gating: If True, only migrate at semantic boundaries
        """
        self.boundary_detector = SemanticBoundaryDetector()
        self.chunk_predictor = ChunkPredictor()
        self.waterline_monitor = WaterlineMonitor(
            hot_high_waterline=hot_high_waterline,
            warm_high_waterline=warm_high_waterline,
            warm_low_waterline=warm_low_waterline
        )
        self.enable_semantic_gating = enable_semantic_gating

        # Track last migration position for debouncing
        self.last_migration_position: Dict[MigrationType, int] = {}

    def is_safe_migration_point(
        self,
        tokens: Optional[List[str]] = None,
        current_position: Optional[int] = None
    ) -> bool:
        """
        Check if current position is safe for migration.

        Args:
            tokens: Token sequence (optional, for semantic checking)
            current_position: Current generation position

        Returns:
            True if safe to migrate
        """
        if not self.enable_semantic_gating:
            return True

        if tokens is None or current_position is None:
            # No semantic info, allow migration
            return True

        # Check if at boundary
        if current_position > 0:
            return self.boundary_detector.is_boundary(tokens, current_position - 1)

        return True

    def evaluate(
        self,
        hot_utilization: float,
        warm_utilization: float,
        cold_utilization: float = 0.0,
        hot_demotion_candidates: Optional[List[int]] = None,
        warm_demotion_candidates: Optional[List[int]] = None,
        warm_promotion_candidates: Optional[List[int]] = None,
        cold_revival_candidates: Optional[List[int]] = None,
        tokens: Optional[List[str]] = None,
        current_position: Optional[int] = None
    ) -> List[MigrationDecision]:
        """
        Evaluate all migration opportunities.

        Args:
            hot_utilization: Hot tier utilization (0.0-1.0)
            warm_utilization: Warm tier utilization (0.0-1.0)
            cold_utilization: Cold tier utilization (0.0-1.0)
            hot_demotion_candidates: Candidates for Hot → Warm
            warm_demotion_candidates: Candidates for Warm → Cold
            warm_promotion_candidates: Candidates for Warm → Hot
            cold_revival_candidates: Candidates for Cold → Warm
            tokens: Token sequence (optional, for semantic gating)
            current_position: Current generation position (optional)

        Returns:
            List of MigrationDecisions, sorted by urgency (highest first)
        """
        # Check semantic safety
        if not self.is_safe_migration_point(tokens, current_position):
            return []

        decisions = []

        # Check Hot tier demotion (highest priority)
        if hot_demotion_candidates:
            decision = self.waterline_monitor.check_hot_tier(
                utilization=hot_utilization,
                demotion_candidates=hot_demotion_candidates
            )
            if decision:
                decisions.append(decision)

        # Check Warm tier demotion
        if warm_demotion_candidates:
            decision = self.waterline_monitor.check_warm_tier_demotion(
                utilization=warm_utilization,
                demotion_candidates=warm_demotion_candidates
            )
            if decision:
                decisions.append(decision)

        # Check Warm tier promotion
        if warm_promotion_candidates:
            decision = self.waterline_monitor.check_warm_tier_promotion(
                utilization=warm_utilization,
                promotion_candidates=warm_promotion_candidates
            )
            if decision:
                decisions.append(decision)

        # Check Cold revival
        if cold_revival_candidates:
            decision = self.waterline_monitor.check_cold_revival(
                warm_utilization=warm_utilization,
                revival_candidates=cold_revival_candidates
            )
            if decision:
                decisions.append(decision)

        # Sort by urgency (highest first)
        decisions.sort(key=lambda d: d.urgency, reverse=True)

        return decisions

    def get_statistics(self) -> Dict[str, any]:
        """
        Get migration trigger statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "hot_high_waterline": self.waterline_monitor.hot_high_waterline,
            "warm_high_waterline": self.waterline_monitor.warm_high_waterline,
            "warm_low_waterline": self.waterline_monitor.warm_low_waterline,
            "enable_semantic_gating": self.enable_semantic_gating,
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"MigrationTrigger("
            f"hot_high={stats['hot_high_waterline']:.1%}, "
            f"warm_high={stats['warm_high_waterline']:.1%}, "
            f"warm_low={stats['warm_low_waterline']:.1%}, "
            f"semantic_gating={stats['enable_semantic_gating']})"
        )
