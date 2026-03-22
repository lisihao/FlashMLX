"""
Unit tests for MigrationTrigger (Task #71)

Tests migration triggering logic including semantic boundaries,
chunk prediction, and waterline monitoring.
"""

import unittest
from flashmlx.cache.migration_trigger import (
    MigrationTrigger,
    MigrationType,
    MigrationDecision,
    SemanticBoundaryDetector,
    ChunkPredictor,
    WaterlineMonitor
)


class TestMigrationDecision(unittest.TestCase):
    """Tests for MigrationDecision dataclass"""

    def test_valid_decision(self):
        """Test valid migration decision"""
        decision = MigrationDecision(
            migration_type=MigrationType.HOT_TO_WARM,
            layer_indices=[0, 1, 2],
            reason="Test migration",
            urgency=0.5
        )

        self.assertEqual(decision.migration_type, MigrationType.HOT_TO_WARM)
        self.assertEqual(decision.layer_indices, [0, 1, 2])
        self.assertEqual(decision.reason, "Test migration")
        self.assertEqual(decision.urgency, 0.5)

    def test_invalid_urgency_too_high(self):
        """Test that urgency > 1.0 raises error"""
        with self.assertRaises(ValueError):
            MigrationDecision(
                migration_type=MigrationType.HOT_TO_WARM,
                layer_indices=[0],
                reason="Test",
                urgency=1.5
            )

    def test_invalid_urgency_too_low(self):
        """Test that urgency < 0.0 raises error"""
        with self.assertRaises(ValueError):
            MigrationDecision(
                migration_type=MigrationType.HOT_TO_WARM,
                layer_indices=[0],
                reason="Test",
                urgency=-0.1
            )


class TestSemanticBoundaryDetector(unittest.TestCase):
    """Tests for SemanticBoundaryDetector"""

    def setUp(self):
        """Set up test detector"""
        self.detector = SemanticBoundaryDetector()

    def test_sentence_ender_detection(self):
        """Test detection of sentence-ending punctuation"""
        tokens = ["Hello", "world", ".", "Next", "sentence"]

        self.assertTrue(self.detector.is_boundary(tokens, 2))  # "."
        self.assertFalse(self.detector.is_boundary(tokens, 0))  # "Hello"
        self.assertFalse(self.detector.is_boundary(tokens, 1))  # "world"

    def test_question_mark_detection(self):
        """Test question mark as boundary"""
        tokens = ["How", "are", "you", "?"]
        self.assertTrue(self.detector.is_boundary(tokens, 3))

    def test_exclamation_mark_detection(self):
        """Test exclamation mark as boundary"""
        tokens = ["Stop", "!"]
        self.assertTrue(self.detector.is_boundary(tokens, 1))

    def test_chinese_punctuation(self):
        """Test Chinese punctuation marks"""
        tokens = ["你好", "。", "谢谢", "！"]
        self.assertTrue(self.detector.is_boundary(tokens, 1))  # "。"
        self.assertTrue(self.detector.is_boundary(tokens, 3))  # "！"

    def test_newline_detection(self):
        """Test newline as boundary"""
        tokens = ["Line", "one", "\n", "Line", "two"]
        self.assertTrue(self.detector.is_boundary(tokens, 2))

    def test_double_newline_detection(self):
        """Test double newline (paragraph break)"""
        tokens = ["Paragraph", "one", "\n\n", "Paragraph", "two"]
        self.assertTrue(self.detector.is_boundary(tokens, 2))

    def test_eos_token_detection(self):
        """Test end-of-sequence tokens"""
        tokens = ["Some", "text", "</s>"]
        self.assertTrue(self.detector.is_boundary(tokens, 2))

    def test_chat_template_end(self):
        """Test chat template end marker"""
        tokens = ["Response", "text", "<|im_end|>"]
        self.assertTrue(self.detector.is_boundary(tokens, 2))

    def test_find_next_boundary(self):
        """Test finding next boundary"""
        tokens = ["Hello", "world", ".", "How", "are", "you", "?"]

        # From start
        self.assertEqual(self.detector.find_next_boundary(tokens, 0), 2)

        # After first boundary
        self.assertEqual(self.detector.find_next_boundary(tokens, 3), 6)

        # No more boundaries
        self.assertIsNone(self.detector.find_next_boundary(tokens, 7))

    def test_get_boundaries(self):
        """Test getting all boundaries"""
        tokens = ["First", "sentence", ".", "Second", "sentence", "!"]
        boundaries = self.detector.get_boundaries(tokens)

        self.assertEqual(len(boundaries), 2)
        self.assertIn(2, boundaries)  # "."
        self.assertIn(5, boundaries)  # "!"

    def test_no_boundaries(self):
        """Test text with no boundaries"""
        tokens = ["no", "punctuation", "here"]
        boundaries = self.detector.get_boundaries(tokens)
        self.assertEqual(len(boundaries), 0)


class TestChunkPredictor(unittest.TestCase):
    """Tests for ChunkPredictor"""

    def setUp(self):
        """Set up test predictor"""
        self.predictor = ChunkPredictor()

    def test_new_chunk_after_period(self):
        """Test new chunk detection after period"""
        self.assertTrue(self.predictor.is_new_chunk(".", "The"))

    def test_new_chunk_after_question(self):
        """Test new chunk after question mark"""
        self.assertTrue(self.predictor.is_new_chunk("?", "What"))

    def test_new_chunk_after_exclamation(self):
        """Test new chunk after exclamation"""
        self.assertTrue(self.predictor.is_new_chunk("!", "Wow"))

    def test_new_chunk_after_newline(self):
        """Test new chunk after newline"""
        self.assertTrue(self.predictor.is_new_chunk("\n", "New"))

    def test_new_chunk_after_eos(self):
        """Test new chunk after EOS token"""
        self.assertTrue(self.predictor.is_new_chunk("</s>", "Next"))

    def test_new_chunk_after_chat_end(self):
        """Test new chunk after chat template end"""
        self.assertTrue(self.predictor.is_new_chunk("<|im_end|>", "Assistant"))

    def test_not_new_chunk_in_sentence(self):
        """Test no new chunk in middle of sentence"""
        self.assertFalse(self.predictor.is_new_chunk("the", "quick"))
        self.assertFalse(self.predictor.is_new_chunk("quick", "brown"))

    def test_not_new_chunk_after_comma(self):
        """Test comma doesn't trigger new chunk"""
        self.assertFalse(self.predictor.is_new_chunk(",", "and"))


class TestWaterlineMonitor(unittest.TestCase):
    """Tests for WaterlineMonitor"""

    def setUp(self):
        """Set up test monitor"""
        self.monitor = WaterlineMonitor(
            hot_high_waterline=0.80,
            warm_high_waterline=0.80,
            warm_low_waterline=0.30
        )

    def test_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.hot_high_waterline, 0.80)
        self.assertEqual(self.monitor.warm_high_waterline, 0.80)
        self.assertEqual(self.monitor.warm_low_waterline, 0.30)

    def test_invalid_hot_waterline_zero(self):
        """Test that hot_high_waterline = 0 raises error"""
        with self.assertRaises(ValueError):
            WaterlineMonitor(hot_high_waterline=0.0)

    def test_invalid_hot_waterline_negative(self):
        """Test that negative hot_high_waterline raises error"""
        with self.assertRaises(ValueError):
            WaterlineMonitor(hot_high_waterline=-0.1)

    def test_hot_tier_below_waterline(self):
        """Test Hot tier below waterline (no action)"""
        decision = self.monitor.check_hot_tier(
            utilization=0.75,
            demotion_candidates=[0, 1]
        )
        self.assertIsNone(decision)

    def test_hot_tier_above_waterline(self):
        """Test Hot tier above waterline (trigger demotion)"""
        decision = self.monitor.check_hot_tier(
            utilization=0.85,
            demotion_candidates=[0, 1, 2]
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.migration_type, MigrationType.HOT_TO_WARM)
        self.assertEqual(decision.layer_indices, [0, 1, 2])
        self.assertGreater(decision.urgency, 0.0)

    def test_hot_tier_urgency_increases(self):
        """Test urgency increases with utilization"""
        decision1 = self.monitor.check_hot_tier(utilization=0.85, demotion_candidates=[0])
        decision2 = self.monitor.check_hot_tier(utilization=0.95, demotion_candidates=[0])

        self.assertGreater(decision2.urgency, decision1.urgency)

    def test_hot_tier_no_candidates(self):
        """Test Hot tier with no candidates"""
        decision = self.monitor.check_hot_tier(
            utilization=0.95,
            demotion_candidates=[]
        )
        self.assertIsNone(decision)

    def test_warm_tier_demotion_below_waterline(self):
        """Test Warm tier demotion below waterline"""
        decision = self.monitor.check_warm_tier_demotion(
            utilization=0.70,
            demotion_candidates=[0]
        )
        self.assertIsNone(decision)

    def test_warm_tier_demotion_above_waterline(self):
        """Test Warm tier demotion above waterline"""
        decision = self.monitor.check_warm_tier_demotion(
            utilization=0.90,
            demotion_candidates=[3, 4]
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.migration_type, MigrationType.WARM_TO_COLD)
        self.assertEqual(decision.layer_indices, [3, 4])

    def test_warm_tier_promotion_high_utilization(self):
        """Test no promotion when Warm is full"""
        decision = self.monitor.check_warm_tier_promotion(
            utilization=0.85,
            promotion_candidates=[5]
        )
        self.assertIsNone(decision)

    def test_warm_tier_promotion_low_utilization(self):
        """Test promotion when Warm has space"""
        decision = self.monitor.check_warm_tier_promotion(
            utilization=0.50,
            promotion_candidates=[5, 6]
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.migration_type, MigrationType.WARM_TO_HOT)
        self.assertEqual(decision.layer_indices, [5, 6])

    def test_cold_revival_no_space(self):
        """Test no revival when Warm is full"""
        decision = self.monitor.check_cold_revival(
            warm_utilization=0.85,
            revival_candidates=[10]
        )
        self.assertIsNone(decision)

    def test_cold_revival_has_space(self):
        """Test revival when Warm has space"""
        decision = self.monitor.check_cold_revival(
            warm_utilization=0.20,
            revival_candidates=[10, 11]
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.migration_type, MigrationType.COLD_TO_WARM)
        self.assertEqual(decision.layer_indices, [10, 11])


class TestMigrationTriggerBasic(unittest.TestCase):
    """Basic functionality tests for MigrationTrigger"""

    def setUp(self):
        """Set up test trigger"""
        self.trigger = MigrationTrigger()

    def test_initialization(self):
        """Test trigger initialization"""
        self.assertIsNotNone(self.trigger.boundary_detector)
        self.assertIsNotNone(self.trigger.chunk_predictor)
        self.assertIsNotNone(self.trigger.waterline_monitor)

    def test_safe_migration_point_no_tokens(self):
        """Test safe migration when no tokens provided"""
        safe = self.trigger.is_safe_migration_point(tokens=None, current_position=None)
        self.assertTrue(safe)

    def test_safe_migration_point_at_boundary(self):
        """Test safe migration at semantic boundary"""
        tokens = ["Hello", "world", ".", "Next"]
        safe = self.trigger.is_safe_migration_point(tokens=tokens, current_position=3)
        self.assertTrue(safe)

    def test_unsafe_migration_point_mid_sentence(self):
        """Test unsafe migration in middle of sentence"""
        tokens = ["Hello", "world", "here"]
        safe = self.trigger.is_safe_migration_point(tokens=tokens, current_position=2)
        self.assertFalse(safe)

    def test_semantic_gating_disabled(self):
        """Test migration always safe when gating disabled"""
        trigger = MigrationTrigger(enable_semantic_gating=False)
        tokens = ["Hello", "world", "here"]
        safe = trigger.is_safe_migration_point(tokens=tokens, current_position=2)
        self.assertTrue(safe)


class TestMigrationTriggerEvaluation(unittest.TestCase):
    """Tests for migration evaluation"""

    def setUp(self):
        """Set up test trigger"""
        self.trigger = MigrationTrigger()

    def test_evaluate_hot_tier_pressure(self):
        """Test evaluation with Hot tier pressure"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.90,
            warm_utilization=0.50,
            hot_demotion_candidates=[0, 1, 2]
        )

        # Should trigger Hot → Warm demotion
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].migration_type, MigrationType.HOT_TO_WARM)
        self.assertEqual(decisions[0].layer_indices, [0, 1, 2])

    def test_evaluate_warm_tier_pressure(self):
        """Test evaluation with Warm tier pressure"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.50,
            warm_utilization=0.90,
            warm_demotion_candidates=[3, 4]
        )

        # Should trigger Warm → Cold demotion
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].migration_type, MigrationType.WARM_TO_COLD)

    def test_evaluate_multiple_decisions(self):
        """Test evaluation with multiple migration opportunities"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.85,
            warm_utilization=0.85,
            hot_demotion_candidates=[0, 1],
            warm_demotion_candidates=[3, 4],
            warm_promotion_candidates=[5]
        )

        # Should have 2 decisions (Hot demotion + Warm demotion)
        # Promotion blocked because Warm is full
        self.assertEqual(len(decisions), 2)

        # Should be sorted by urgency (Hot demotion first)
        self.assertEqual(decisions[0].migration_type, MigrationType.HOT_TO_WARM)

    def test_evaluate_promotion_opportunity(self):
        """Test promotion when conditions are right"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.50,
            warm_utilization=0.40,  # Below high waterline
            warm_promotion_candidates=[5, 6]
        )

        # Should trigger Warm → Hot promotion
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].migration_type, MigrationType.WARM_TO_HOT)

    def test_evaluate_revival_opportunity(self):
        """Test revival when Warm has space"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.50,
            warm_utilization=0.20,  # Well below threshold
            cold_revival_candidates=[10, 11, 12]
        )

        # Should trigger Cold → Warm revival
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].migration_type, MigrationType.COLD_TO_WARM)

    def test_evaluate_blocked_by_semantic_boundary(self):
        """Test that migrations are blocked outside boundaries"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.95,
            warm_utilization=0.50,
            hot_demotion_candidates=[0],
            tokens=["Hello", "world", "here"],  # No boundary
            current_position=2
        )

        # Should be blocked (not at boundary)
        self.assertEqual(len(decisions), 0)

    def test_evaluate_allowed_at_boundary(self):
        """Test that migrations are allowed at boundaries"""
        decisions = self.trigger.evaluate(
            hot_utilization=0.95,
            warm_utilization=0.50,
            hot_demotion_candidates=[0],
            tokens=["Hello", "world", "."],  # Boundary at position 2
            current_position=3
        )

        # Should be allowed
        self.assertGreater(len(decisions), 0)

    def test_urgency_sorting(self):
        """Test that decisions are sorted by urgency"""
        # Create conditions for multiple migrations
        decisions = self.trigger.evaluate(
            hot_utilization=0.95,  # High urgency
            warm_utilization=0.85,  # Medium urgency
            hot_demotion_candidates=[0],
            warm_demotion_candidates=[3],
            warm_promotion_candidates=[5]  # Low urgency
        )

        # Hot demotion should be first (highest urgency)
        self.assertGreater(decisions[0].urgency, decisions[1].urgency)


class TestMigrationTriggerStatistics(unittest.TestCase):
    """Tests for statistics and representation"""

    def setUp(self):
        """Set up test trigger"""
        self.trigger = MigrationTrigger()

    def test_statistics_structure(self):
        """Test statistics dictionary structure"""
        stats = self.trigger.get_statistics()

        required_keys = [
            "hot_high_waterline",
            "warm_high_waterline",
            "warm_low_waterline",
            "enable_semantic_gating"
        ]

        for key in required_keys:
            self.assertIn(key, stats)

    def test_repr(self):
        """Test __repr__ method"""
        repr_str = repr(self.trigger)

        self.assertIn("MigrationTrigger", repr_str)
        self.assertIn("hot_high=", repr_str)
        self.assertIn("warm_high=", repr_str)
        self.assertIn("semantic_gating=", repr_str)


if __name__ == "__main__":
    unittest.main()
