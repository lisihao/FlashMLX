"""
Unit tests for PinnedControlState (Task #69)

Tests control channel detection and protection from eviction.
"""

import unittest

from flashmlx.cache.pinned_control_state import (
    PinnedControlState,
    ControlChannel,
    ControlChannelType
)


class TestControlChannel(unittest.TestCase):
    """Tests for ControlChannel dataclass"""

    def test_valid_control_channel(self):
        """Test valid control channel creation"""
        channel = ControlChannel(
            channel_type=ControlChannelType.LANGUAGE,
            position=0,
            importance=0.95,
            content="请用中文回答"
        )

        self.assertEqual(channel.channel_type, ControlChannelType.LANGUAGE)
        self.assertEqual(channel.position, 0)
        self.assertEqual(channel.importance, 0.95)
        self.assertEqual(channel.content, "请用中文回答")

    def test_invalid_importance_too_low(self):
        """Test that importance < 0 raises error"""
        with self.assertRaises(ValueError):
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=-0.1,
                content="test"
            )

    def test_invalid_importance_too_high(self):
        """Test that importance > 1 raises error"""
        with self.assertRaises(ValueError):
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=1.5,
                content="test"
            )

    def test_invalid_position_negative(self):
        """Test that negative position raises error"""
        with self.assertRaises(ValueError):
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=-1,
                importance=0.8,
                content="test"
            )


class TestPinnedControlStateBasic(unittest.TestCase):
    """Basic functionality tests for PinnedControlState"""

    def setUp(self):
        """Set up test manager"""
        self.manager = PinnedControlState(max_pinned_positions=100)

    def test_initialization(self):
        """Test PinnedControlState initialization"""
        self.assertEqual(self.manager.max_pinned_positions, 100)
        self.assertEqual(len(self.manager.pinned_positions), 0)
        self.assertEqual(len(self.manager.control_channels), 0)

    def test_patterns_initialized(self):
        """Test that detection patterns are initialized"""
        # Should have patterns for all channel types
        expected_types = [
            ControlChannelType.LANGUAGE,
            ControlChannelType.FORMAT,
            ControlChannelType.THINK_MODE,
            ControlChannelType.DELIMITER
        ]

        for channel_type in expected_types:
            self.assertIn(channel_type, self.manager.patterns)
            self.assertGreater(len(self.manager.patterns[channel_type]), 0)


class TestControlChannelDetection(unittest.TestCase):
    """Tests for control channel detection"""

    def setUp(self):
        """Set up test manager"""
        self.manager = PinnedControlState()

    def test_detect_language_chinese(self):
        """Test detection of Chinese language instruction"""
        text = "请用中文回答这个问题。"

        channels = self.manager.detect_control_channels(text)

        # Should detect language channel
        language_channels = [c for c in channels if c.channel_type == ControlChannelType.LANGUAGE]
        self.assertGreater(len(language_channels), 0)

        # Should have high importance (at start of text)
        self.assertGreater(language_channels[0].importance, 0.8)

    def test_detect_language_english(self):
        """Test detection of English language instruction"""
        text = "Please answer in English."

        channels = self.manager.detect_control_channels(text)

        language_channels = [c for c in channels if c.channel_type == ControlChannelType.LANGUAGE]
        self.assertGreater(len(language_channels), 0)

    def test_detect_format_list(self):
        """Test detection of format instruction (list)"""
        text = "请使用列表格式输出结果。"

        channels = self.manager.detect_control_channels(text)

        format_channels = [c for c in channels if c.channel_type == ControlChannelType.FORMAT]
        self.assertGreater(len(format_channels), 0)

    def test_detect_format_table(self):
        """Test detection of format instruction (table)"""
        text = "Use table format for the output."

        channels = self.manager.detect_control_channels(text)

        format_channels = [c for c in channels if c.channel_type == ControlChannelType.FORMAT]
        self.assertGreater(len(format_channels), 0)

    def test_detect_think_mode_tags(self):
        """Test detection of think mode tags"""
        text = "<think>Let me analyze this problem...</think>"

        channels = self.manager.detect_control_channels(text)

        think_channels = [c for c in channels if c.channel_type == ControlChannelType.THINK_MODE]
        # Should detect both <think> and </think>
        self.assertEqual(len(think_channels), 2)

    def test_detect_delimiter(self):
        """Test detection of delimiter"""
        text = "Section 1\n---\nSection 2"

        # Use lower threshold to detect delimiter (base importance 0.5)
        channels = self.manager.detect_control_channels(text, importance_threshold=0.3)

        delimiter_channels = [c for c in channels if c.channel_type == ControlChannelType.DELIMITER]
        self.assertGreater(len(delimiter_channels), 0)

    def test_detect_multiple_channel_types(self):
        """Test detection of multiple channel types"""
        # Use explicit newlines to avoid indentation in multiline string
        text = "请用中文回答。<think>分析问题...</think>使用列表格式。\n---\n答案是..."

        # Use lower threshold to detect delimiter
        channels = self.manager.detect_control_channels(text, importance_threshold=0.3)

        # Should detect language, think mode, format, delimiter
        channel_types = {c.channel_type for c in channels}
        self.assertIn(ControlChannelType.LANGUAGE, channel_types)
        self.assertIn(ControlChannelType.THINK_MODE, channel_types)
        self.assertIn(ControlChannelType.FORMAT, channel_types)
        self.assertIn(ControlChannelType.DELIMITER, channel_types)

    def test_importance_threshold(self):
        """Test that importance threshold filters channels"""
        text = "请用中文回答。" + "X" * 1000  # Language at start, then padding

        # High threshold
        channels_high = self.manager.detect_control_channels(text, importance_threshold=0.9)
        # Low threshold
        channels_low = self.manager.detect_control_channels(text, importance_threshold=0.3)

        # High threshold should detect fewer channels
        self.assertLessEqual(len(channels_high), len(channels_low))


class TestImportanceCalculation(unittest.TestCase):
    """Tests for importance calculation"""

    def setUp(self):
        """Set up test manager"""
        self.manager = PinnedControlState()

    def test_language_has_highest_importance(self):
        """Test that language channels have highest base importance"""
        # Language at position 0
        importance_lang = self.manager._calculate_importance(
            channel_type=ControlChannelType.LANGUAGE,
            position=0,
            text_length=100
        )

        # Format at position 0
        importance_fmt = self.manager._calculate_importance(
            channel_type=ControlChannelType.FORMAT,
            position=0,
            text_length=100
        )

        self.assertGreater(importance_lang, importance_fmt)

    def test_position_affects_importance(self):
        """Test that earlier positions have higher importance"""
        text_length = 1000

        # Language at start
        importance_start = self.manager._calculate_importance(
            channel_type=ControlChannelType.LANGUAGE,
            position=0,
            text_length=text_length
        )

        # Language at end
        importance_end = self.manager._calculate_importance(
            channel_type=ControlChannelType.LANGUAGE,
            position=900,
            text_length=text_length
        )

        self.assertGreater(importance_start, importance_end)

    def test_importance_bounded(self):
        """Test that importance is always in [0, 1]"""
        for channel_type in ControlChannelType:
            for position in [0, 50, 100]:
                importance = self.manager._calculate_importance(
                    channel_type=channel_type,
                    position=position,
                    text_length=100
                )

                self.assertGreaterEqual(importance, 0.0)
                self.assertLessEqual(importance, 1.0)


class TestPinning(unittest.TestCase):
    """Tests for pinning operations"""

    def setUp(self):
        """Set up test manager"""
        self.manager = PinnedControlState(max_pinned_positions=10)

    def test_pin_channels(self):
        """Test pinning channels"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=0.95,
                content="中文"
            ),
            ControlChannel(
                channel_type=ControlChannelType.FORMAT,
                position=10,
                importance=0.8,
                content="列表"
            )
        ]

        self.manager.pin_channels(channels)

        self.assertEqual(self.manager.get_pinned_count(), 2)
        self.assertTrue(self.manager.is_pinned(0))
        self.assertTrue(self.manager.is_pinned(10))

    def test_pin_channels_respects_max(self):
        """Test that pinning respects max_pinned_positions"""
        # Create 15 channels
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=i,
                importance=0.9 - i * 0.01,  # Decreasing importance
                content=f"channel_{i}"
            )
            for i in range(15)
        ]

        self.manager.pin_channels(channels)

        # Should only pin top 10
        self.assertEqual(self.manager.get_pinned_count(), 10)

    def test_pin_channels_by_importance(self):
        """Test that channels are pinned by importance order"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=0.5,  # Low importance
                content="low"
            ),
            ControlChannel(
                channel_type=ControlChannelType.FORMAT,
                position=1,
                importance=0.95,  # High importance
                content="high"
            ),
            ControlChannel(
                channel_type=ControlChannelType.THINK_MODE,
                position=2,
                importance=0.7,  # Medium importance
                content="medium"
            )
        ]

        self.manager.pin_channels(channels)

        # All 3 should be pinned (within max)
        self.assertEqual(self.manager.get_pinned_count(), 3)

        # Check that high importance is pinned
        self.assertTrue(self.manager.is_pinned(1))

    def test_unpin(self):
        """Test unpinning a position"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=0.95,
                content="test"
            )
        ]

        self.manager.pin_channels(channels)
        self.assertTrue(self.manager.is_pinned(0))

        # Unpin
        success = self.manager.unpin(0)

        self.assertTrue(success)
        self.assertFalse(self.manager.is_pinned(0))
        self.assertEqual(self.manager.get_pinned_count(), 0)

    def test_unpin_nonexistent(self):
        """Test unpinning non-existent position"""
        success = self.manager.unpin(999)

        self.assertFalse(success)

    def test_get_pinned_positions(self):
        """Test getting all pinned positions"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=pos,
                importance=0.9,
                content=f"channel_{pos}"
            )
            for pos in [5, 2, 8, 1]
        ]

        self.manager.pin_channels(channels)

        positions = self.manager.get_pinned_positions()

        # Should be sorted
        self.assertEqual(positions, [1, 2, 5, 8])


class TestChannelQueries(unittest.TestCase):
    """Tests for channel query operations"""

    def setUp(self):
        """Set up test manager"""
        self.manager = PinnedControlState()

    def test_get_channels_by_type(self):
        """Test getting channels by type"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=0.95,
                content="lang1"
            ),
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=10,
                importance=0.9,
                content="lang2"
            ),
            ControlChannel(
                channel_type=ControlChannelType.FORMAT,
                position=20,
                importance=0.8,
                content="fmt1"
            )
        ]

        self.manager.pin_channels(channels)

        # Get language channels
        language_channels = self.manager.get_channels_by_type(ControlChannelType.LANGUAGE)

        self.assertEqual(len(language_channels), 2)
        for channel in language_channels:
            self.assertEqual(channel.channel_type, ControlChannelType.LANGUAGE)

        # Get format channels
        format_channels = self.manager.get_channels_by_type(ControlChannelType.FORMAT)
        self.assertEqual(len(format_channels), 1)

    def test_clear(self):
        """Test clearing all pinned positions"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=i,
                importance=0.9,
                content=f"channel_{i}"
            )
            for i in range(5)
        ]

        self.manager.pin_channels(channels)
        self.assertEqual(self.manager.get_pinned_count(), 5)

        # Clear
        self.manager.clear()

        self.assertEqual(self.manager.get_pinned_count(), 0)
        self.assertEqual(len(self.manager.control_channels), 0)


class TestStatistics(unittest.TestCase):
    """Tests for statistics tracking"""

    def setUp(self):
        """Set up test manager"""
        self.manager = PinnedControlState(max_pinned_positions=20)

    def test_statistics_structure(self):
        """Test statistics dictionary structure"""
        stats = self.manager.get_statistics()

        required_keys = [
            "total_pinned",
            "max_pinned",
            "utilization",
            "channels_by_type",
            "avg_importance"
        ]

        for key in required_keys:
            self.assertIn(key, stats)

    def test_statistics_empty(self):
        """Test statistics when empty"""
        stats = self.manager.get_statistics()

        self.assertEqual(stats["total_pinned"], 0)
        self.assertEqual(stats["max_pinned"], 20)
        self.assertEqual(stats["utilization"], 0.0)
        self.assertEqual(stats["avg_importance"], 0.0)

    def test_statistics_with_channels(self):
        """Test statistics with pinned channels"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=0.9,
                content="lang"
            ),
            ControlChannel(
                channel_type=ControlChannelType.FORMAT,
                position=10,
                importance=0.7,
                content="fmt"
            )
        ]

        self.manager.pin_channels(channels)

        stats = self.manager.get_statistics()

        self.assertEqual(stats["total_pinned"], 2)
        self.assertEqual(stats["utilization"], 2 / 20)
        self.assertAlmostEqual(stats["avg_importance"], 0.8)  # (0.9 + 0.7) / 2

    def test_statistics_channels_by_type(self):
        """Test channels_by_type in statistics"""
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=0,
                importance=0.9,
                content="lang1"
            ),
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=10,
                importance=0.85,
                content="lang2"
            ),
            ControlChannel(
                channel_type=ControlChannelType.FORMAT,
                position=20,
                importance=0.8,
                content="fmt"
            )
        ]

        self.manager.pin_channels(channels)

        stats = self.manager.get_statistics()

        self.assertEqual(stats["channels_by_type"]["language"], 2)
        self.assertEqual(stats["channels_by_type"]["format"], 1)
        self.assertEqual(stats["channels_by_type"]["think_mode"], 0)


class TestRepr(unittest.TestCase):
    """Tests for string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        manager = PinnedControlState(max_pinned_positions=50)

        # Pin some channels
        channels = [
            ControlChannel(
                channel_type=ControlChannelType.LANGUAGE,
                position=i,
                importance=0.9,
                content=f"channel_{i}"
            )
            for i in range(10)
        ]
        manager.pin_channels(channels)

        repr_str = repr(manager)

        # Should contain key information
        self.assertIn("PinnedControlState", repr_str)
        self.assertIn("pinned=", repr_str)
        self.assertIn("10/50", repr_str)  # 10 pinned out of 50 max
        self.assertIn("avg_importance=", repr_str)


if __name__ == "__main__":
    unittest.main()
