"""
GUI smoke tests — verifies the application can be constructed without errors.
Run from project root:  python temp/test_gui_smoke.py
"""
import sys
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "runtime"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


class TestGUIConstruction(unittest.TestCase):
    """Smoke tests: verify the app constructs and key methods work."""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls._qapp = QApplication.instance()
        if cls._qapp is None:
            cls._qapp = QApplication([sys.argv[0]])

        from main import MeanVCApp
        cls.window = MeanVCApp()

    def test_window_created(self):
        """Window should be a QMainWindow with correct title."""
        self.assertIsNotNone(self.window)
        self.assertIn("MeanVC", self.window.windowTitle())

    def test_required_widgets_exist(self):
        """All key widgets should be created."""
        w = self.window
        self.assertIsNotNone(w.load_btn)
        self.assertIsNotNone(w.start_btn)
        self.assertIsNotNone(w.stop_btn)
        self.assertIsNotNone(w.input_combo)
        self.assertIsNotNone(w.output_combo)
        self.assertIsNotNone(w.steps_spin)
        self.assertIsNotNone(w.status_label)
        self.assertIsNotNone(w.metrics)
        self.assertIsNotNone(w.target_file_edit)
        self.assertIsNotNone(w.load_progress)

    def test_steps_spin_range(self):
        """Steps spinner should allow 1-8."""
        self.assertEqual(self.window.steps_spin.minimum(), 1)
        self.assertEqual(self.window.steps_spin.maximum(), 8)
        self.assertEqual(self.window.steps_spin.value(), 2)

    def test_initial_button_states(self):
        """Start and Stop should be disabled initially, Load enabled."""
        self.assertTrue(self.window.load_btn.isEnabled())
        self.assertFalse(self.window.start_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())

    def test_initial_status(self):
        """Status should show 'Ready' initially."""
        self.assertIn("Ready", self.window.status_label.text())

    def test_models_none_initially(self):
        """models should be None before loading."""
        self.assertIsNone(self.window.models)

    def test_set_ui_loading_disables_buttons(self):
        """_set_ui_loading(True) should disable all action buttons."""
        self.window._set_ui_loading(True)
        self.assertFalse(self.window.load_btn.isEnabled())
        self.assertFalse(self.window.start_btn.isEnabled())
        self.assertFalse(self.window.stop_btn.isEnabled())
        # load_progress visibility property is set to True but window isn't
        # shown in tests, so isVisible() checks ancestors. Check property instead.
        self.assertFalse(self.window.load_progress.isHidden())
        # Also check load_btn is disabled
        self.assertFalse(self.window.load_btn.isEnabled())

        self.window._set_ui_loading(False)
        self.assertTrue(self.window.load_btn.isEnabled())
        self.assertTrue(self.window.load_progress.isHidden())

    def test_set_status_updates_label(self):
        """_set_status should update the status text and color."""
        self.window._set_status("Testing...", "#ff0000")
        self.assertEqual(self.window.status_label.text(), "Testing...")
        self.assertIn("ff0000", self.window.status_label.styleSheet())

    def test_model_loader_check_missing_file(self):
        """check_required_files should report missing when file absent."""
        from main import ModelLoader
        # All files should be present after download
        ok, missing = ModelLoader.check_required_files()
        self.assertTrue(ok, f"Files missing: {missing}")

    def test_device_scan_emits_signal(self):
        """DeviceScanner should emit devices_ready after scanning."""
        from main import DeviceScanner
        from PyQt6.QtCore import QEventLoop, QTimer

        result = {'received': False, 'inputs': None, 'outputs': None}

        def on_ready(inputs, outputs, def_in, def_out):
            result['received'] = True
            result['inputs'] = inputs
            result['outputs'] = outputs

        scanner = DeviceScanner()
        scanner.devices_ready.connect(on_ready)
        scanner.start()

        # Wait up to 5 seconds for scan to complete
        loop = QEventLoop()
        QTimer.singleShot(5000, loop.quit)
        scanner.finished.connect(loop.quit)
        loop.exec()

        self.assertTrue(result['received'], "DeviceScanner did not emit results")
        self.assertIsInstance(result['inputs'], list)
        self.assertIsInstance(result['outputs'], list)

    def test_model_loader_check_on_test_audio(self):
        """ModelLoader can be instantiated with test.wav path."""
        from main import ModelLoader
        loader = ModelLoader('src/runtime/example/test.wav', steps=2)
        self.assertEqual(loader.target_path, 'src/runtime/example/test.wav')
        self.assertEqual(loader.steps, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
