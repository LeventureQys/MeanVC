"""
Unit tests for VCWorker state management and MetricsPanel.
Run from project root:  python temp/test_vcworker.py
"""
import sys
import os
import time
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "runtime"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import torch

from main import VCWorker, MetricsPanel


class TestVCWorkerState(unittest.TestCase):
    """Test VCWorker state management (model-independent)."""

    def setUp(self):
        self.worker = VCWorker()

    def tearDown(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)

    def test_initial_state(self):
        """Worker should start with clean state."""
        self.assertIsNone(self.worker.models)
        self.assertIsNone(self.worker.input_device_id)
        self.assertIsNone(self.worker.output_device_id)
        self.assertFalse(self.worker._running)
        self.assertFalse(self.worker._pause)
        self.assertIsNone(self.worker.samples_cache)
        self.assertIsNone(self.worker.vc_cache)
        self.assertIsNone(self.worker.vc_kv_cache)
        self.assertEqual(self.worker.chunk_counter, 0)

    def test_set_models(self):
        """set_models should store the models dict."""
        models = {'vc': 'fake_vc', 'asr': 'fake_asr'}
        self.worker.set_models(models)
        self.assertEqual(self.worker.models['vc'], 'fake_vc')

    def test_set_devices(self):
        """set_devices should store device IDs."""
        self.worker.set_devices(3, 5)
        self.assertEqual(self.worker.input_device_id, 3)
        self.assertEqual(self.worker.output_device_id, 5)

    def test_init_cache_resets_all_state(self):
        """init_cache should reset all internal state variables."""
        # First set some non-default values
        self.worker.samples_cache = np.ones(100)
        self.worker.vc_offset = 999
        self.worker.chunk_counter = 42
        self.worker.encoder_output_cache = torch.ones(1, 5, 256)
        self.worker.vc_kv_cache = [('k', 'v')]

        self.worker.init_cache()

        self.assertIsNone(self.worker.samples_cache)
        self.assertIsNone(self.worker.encoder_output_cache)
        self.assertIsNone(self.worker.vc_cache)
        self.assertIsNone(self.worker.vc_kv_cache)
        self.assertIsNone(self.worker.last_wav)
        self.assertEqual(self.worker.vc_offset, 0)
        self.assertEqual(self.worker.asr_offset, 0)
        self.assertEqual(self.worker.chunk_counter, 0)
        self.assertIsNone(self.worker.vocoder_cache)
        self.assertTrue(self.worker.need_extra_data)

    def test_reset_cache_partial(self):
        """reset_cache (periodic) should move offsets but not clear caches."""
        self.worker.init_cache()
        self.worker.reset_cache()
        # Only offsets are reset, not the caches
        self.assertEqual(self.worker.asr_offset, 20)
        self.assertEqual(self.worker.vc_offset, 120)

    def test_stop_sets_running_false(self):
        """stop() should set _running to False."""
        self.worker._running = True
        self.worker.stop()
        self.assertFalse(self.worker._running)

    def test_pause_resume(self):
        """Pause and resume should toggle _pause flag."""
        self.assertFalse(self.worker._pause)
        self.worker.pause()
        self.assertTrue(self.worker._pause)
        self.worker.resume()
        self.assertFalse(self.worker._pause)

    def test_multiple_init_cache_idempotent(self):
        """Repeated init_cache calls should be safe."""
        self.worker.init_cache()
        self.worker.init_cache()
        self.worker.init_cache()
        self.assertIsNone(self.worker.vc_cache)

    def test_need_extra_data_flag(self):
        """need_extra_data should start True."""
        self.assertTrue(self.worker.need_extra_data)
        self.worker.need_extra_data = False
        self.assertFalse(self.worker.need_extra_data)
        # init_cache should reset it
        self.worker.init_cache()
        self.assertTrue(self.worker.need_extra_data)


class TestMetricsPanel(unittest.TestCase):
    """Test the rolling-average metrics panel logic."""

    @classmethod
    def setUpClass(cls):
        # MetricsPanel is a QWidget — requires QApplication
        from PyQt6.QtWidgets import QApplication
        cls._qapp = QApplication.instance()
        if cls._qapp is None:
            cls._qapp = QApplication([sys.argv[0]])

    def setUp(self):
        self.panel = MetricsPanel()

    def test_initial_state(self):
        """Metrics panel should start with '--' placeholders."""
        self.assertIn("--", self.panel.chunk_time.value_label.text())
        self.assertIn("--", self.panel.total_latency.value_label.text())
        self.assertIn("--", self.panel.memory.value_label.text())
        self.assertIn("--", self.panel.realtime_factor.value_label.text())
        self.assertIn("--", self.panel.chunk_rate.value_label.text())

    def test_reset_clears_state(self):
        """reset() should clear all internal state and display placeholders."""
        self.panel.update_metrics(50.0, 70.0, 500.0)
        self.panel.update_metrics(55.0, 75.0, 510.0)
        self.assertTrue(len(self.panel._chunk_times) > 0)

        self.panel.reset()
        self.assertEqual(len(self.panel._chunk_times), 0)
        self.assertEqual(len(self.panel._chunk_rate_times), 0)
        self.assertIn("--", self.panel.chunk_time.value_label.text())

    def test_update_metrics_accumulates_samples(self):
        """Each update should add to rolling samples."""
        for i in range(10):
            self.panel.update_metrics(40.0 + i, 60.0 + i, 500.0 + i)
        self.assertEqual(len(self.panel._chunk_times), 10)

    def test_rolling_window_capped_at_50(self):
        """Rolling window should not exceed 50 samples."""
        for i in range(100):
            self.panel.update_metrics(50.0, 70.0, 500.0)
        self.assertLessEqual(len(self.panel._chunk_times), 50)

    def test_display_shows_number_not_placeholder(self):
        """After updates, display should show numbers."""
        self.panel.update_metrics(45.2, 62.1, 523.0)
        self.assertNotIn("--", self.panel.chunk_time.value_label.text())
        self.assertNotIn("--", self.panel.total_latency.value_label.text())
        self.assertNotIn("--", self.panel.memory.value_label.text())

    def test_zero_latency(self):
        """Zero latency should not crash (edge case)."""
        self.panel.update_metrics(0.0, 0.0, 0.0)
        self.assertNotIn("--", self.panel.chunk_time.value_label.text())

    def test_negative_latency_handled(self):
        """Negative latency (should not happen, but test robustness)."""
        self.panel.update_metrics(-10.0, -5.0, 300.0)
        # Just verify it doesn't crash
        self.assertIsNotNone(self.panel.chunk_time.value_label.text())

    def test_large_latency_values(self):
        """Very large latency values should display correctly."""
        self.panel.update_metrics(9999.9, 15000.0, 8192.0)
        self.assertIn("9999", self.panel.chunk_time.value_label.text())
        self.assertIn("8192", self.panel.memory.value_label.text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
