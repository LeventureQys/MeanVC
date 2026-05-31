"""
Unit tests for core utilities and model loading of MeanVC GUI.
Run from project root:  python temp/test_core.py
"""
import sys
import os
import time
import unittest

# Ensure we run from project root and can import main.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "runtime"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import torch

from main import (
    MelSpectrogramFeatures,
    extract_fbanks,
    ModelLoader,
    MetricsPanel,
)


class TestMelSpectrogramFeatures(unittest.TestCase):
    """Test mel spectrogram extraction used in prompt mel computation."""

    def setUp(self):
        self.mel_extract = MelSpectrogramFeatures(
            sample_rate=16000, n_fft=1024, win_size=640,
            hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True
        )

    def test_output_shape(self):
        """Mel spectrogram should have shape (1, 80, T) for (1, samples) input."""
        # 1 second of silence at 16 kHz
        audio = torch.zeros(1, 16000)
        mel = self.mel_extract(audio)
        self.assertEqual(mel.ndim, 3)
        self.assertEqual(mel.shape[1], 80)  # n_mels
        self.assertGreater(mel.shape[2], 1)  # time frames

    def test_output_range(self):
        """Mel spectrogram should be normalized to [-1, 1]."""
        audio = torch.randn(1, 16000) * 0.1
        mel = self.mel_extract(audio)
        self.assertTrue(torch.all(mel >= -1.0))
        self.assertTrue(torch.all(mel <= 1.0))

    def test_different_lengths(self):
        """Should handle different audio lengths."""
        for length in [8000, 16000, 32000, 48000]:
            audio = torch.randn(1, length)
            mel = self.mel_extract(audio)
            self.assertEqual(mel.shape[1], 80)
            # With center=True stft, frames ≈ length/hop_length + 1
            expected_frames = length // 160 + 1
            self.assertAlmostEqual(mel.shape[2], expected_frames, delta=3)

    def test_caching(self):
        """Mel basis and hann window should be cached after first call."""
        self.assertEqual(len(self.mel_extract.mel_basis), 0)
        self.assertEqual(len(self.mel_extract.hann_window), 0)
        audio = torch.randn(1, 8000)
        self.mel_extract(audio)
        self.assertGreater(len(self.mel_extract.mel_basis), 0)
        self.assertGreater(len(self.mel_extract.hann_window), 0)

    def test_empty_audio_raises(self):
        """Empty audio should raise an error."""
        audio = torch.zeros(1, 0)
        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            self.mel_extract(audio)


class TestExtractFbanks(unittest.TestCase):
    """Test Kaldi-compatible fbank extraction."""

    def test_output_shape(self):
        """Fbanks should have shape (1, T, 80)."""
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        fbanks = extract_fbanks(audio, frame_shift=10)
        self.assertEqual(fbanks.ndim, 3)
        self.assertEqual(fbanks.shape[2], 80)     # mel bins
        self.assertGreater(fbanks.shape[1], 1)     # time frames

    def test_output_is_float32(self):
        """Output should be float32."""
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        fbanks = extract_fbanks(audio, frame_shift=10)
        self.assertEqual(fbanks.dtype, torch.float32)

    def test_longer_audio(self):
        """Longer audio should produce proportionally more frames."""
        audio_1s = np.random.randn(16000).astype(np.float32) * 0.01
        audio_2s = np.random.randn(32000).astype(np.float32) * 0.01

        fbanks_1s = extract_fbanks(audio_1s, frame_shift=10)
        fbanks_2s = extract_fbanks(audio_2s, frame_shift=10)

        # 2s should have roughly 2x the frames (not exactly due to padding)
        ratio = fbanks_2s.shape[1] / fbanks_1s.shape[1]
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)


class TestModelLoaderChecks(unittest.TestCase):
    """Test the static file-existence check."""

    def test_required_files_exist(self):
        """All required model files should be present on disk."""
        ok, missing = ModelLoader.check_required_files()

        # Print helpful diagnostic
        if not ok:
            print("\n[WARNING] Some model files are missing:")
            for m in missing:
                print(m)

        self.assertTrue(ok, f"Missing model files:\n" + "\n".join(missing))

    def test_check_returns_bool_and_list(self):
        """check_required_files should return a bool and a list."""
        ok, missing = ModelLoader.check_required_files()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(missing, list)


# ═══════════════════════════════════════════════════════════════════════
# Integration tests (make sure to run with -k "integration" or "all")
# ═══════════════════════════════════════════════════════════════════════

@unittest.skipUnless(
    os.path.exists("src/ckpt/fastu2++.pt"),
    "ASR model not found — run download_ckpt.py first"
)
class TestModelLoading(unittest.TestCase):
    """Test loading of each model component."""

    @classmethod
    def setUpClass(cls):
        from speaker_verification.verification import init_model as init_sv_model

        # Load all models once for the test class
        cls.sv_model = init_sv_model(
            'wavlm_large',
            'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth'
        )
        cls.sv_model.eval()
        cls.asr = torch.jit.load('src/ckpt/fastu2++.pt')
        cls.vc = torch.jit.load('src/ckpt/meanvc_200ms.pt')
        cls.vocoder = torch.jit.load('src/ckpt/vocos.pt')

    def test_sv_model_loaded(self):
        """Speaker verification model should be loaded and in eval mode."""
        self.assertFalse(self.sv_model.training)

    def test_asr_model_is_jit(self):
        """ASR model should be a TorchScript module."""
        self.assertIsInstance(self.asr, torch.jit.ScriptModule)

    def test_vc_model_is_jit(self):
        """VC model should be a TorchScript module."""
        self.assertIsInstance(self.vc, torch.jit.ScriptModule)

    def test_vocoder_is_jit(self):
        """Vocoder should be a TorchScript module."""
        self.assertIsInstance(self.vocoder, torch.jit.ScriptModule)

    def test_speaker_embedding_shape(self):
        """Speaker embedding from test.wav should be (1, 256)."""
        import librosa
        wav, _ = librosa.load('src/runtime/example/test.wav', sr=16000)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        spk_emb = self.sv_model(wav_tensor)
        self.assertEqual(spk_emb.shape, (1, 256))
        self.assertEqual(spk_emb.dtype, torch.float32)

    def test_prompt_mel_shape(self):
        """Prompt mel from test.wav should have shape (1, T, 80) after transpose."""
        import librosa
        mel_extract = MelSpectrogramFeatures()

        wav, _ = librosa.load('src/runtime/example/test.wav', sr=16000)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        prompt_mel = mel_extract(wav_tensor).transpose(1, 2)
        self.assertEqual(prompt_mel.ndim, 3)
        self.assertEqual(prompt_mel.shape[2], 80)
        self.assertGreater(prompt_mel.shape[1], 1)

    def test_vocoder_decode(self):
        """Vocoder should decode mel spectrogram to audio."""
        import librosa
        mel_extract = MelSpectrogramFeatures()

        wav, _ = librosa.load('src/runtime/example/test.wav', sr=16000)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        mel = mel_extract(wav_tensor)
        mel = (mel + 1) / 2  # un-normalize
        decoded = self.vocoder.decode(mel).squeeze()
        self.assertGreater(decoded.numel(), 0)
        self.assertEqual(decoded.ndim, 1)  # mono

    def test_full_inference_pipeline(self):
        """End-to-end single chunk inference."""
        import librosa

        torch.set_num_threads(1)

        # Prepare target speaker features
        wav, _ = librosa.load('src/runtime/example/test.wav', sr=16000)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)

        mel_extract = MelSpectrogramFeatures()
        spk_emb = self.sv_model(wav_tensor)
        prompt_mel = mel_extract(wav_tensor).transpose(1, 2)

        # Simulate a chunk of audio (CHUNK + 720 extra samples)
        CHUNK = 160 * 20  # 3200 samples
        np.random.seed(42)
        torch.manual_seed(42)

        samples = np.random.randn(CHUNK + 720).astype(np.float32) * 0.01
        fbanks = extract_fbanks(samples, frame_shift=10).float()

        # ASR encoder chunk
        att_cache = torch.zeros((0, 0, 0, 0), device='cpu')
        cnn_cache = torch.zeros((0, 0, 0, 0), device='cpu')
        encoder_output, att_cache, cnn_cache = self.asr.forward_encoder_chunk(
            fbanks, 0, 10, att_cache, cnn_cache)

        # Prepare BN features
        encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
        vc_chunk = 20
        encoder_up = encoder_output.transpose(1, 2)
        encoder_up = torch.nn.functional.interpolate(
            encoder_up, size=vc_chunk + 1, mode='linear', align_corners=True)
        encoder_up = encoder_up.transpose(1, 2)[:, 1:, :]

        # VC inference (2 denoising steps)
        x = torch.randn(1, encoder_up.shape[1], 80, dtype=encoder_up.dtype)
        timesteps = torch.tensor([1.0, 0.8, 0.0])

        for i in range(2):
            t = timesteps[i]
            r = timesteps[i + 1]
            t_tensor = torch.full((1,), t)
            r_tensor = torch.full((1,), r)
            u, _ = self.vc(x, t_tensor, r_tensor, cache=None, cond=encoder_up,
                           spks=spk_emb, prompts=prompt_mel, offset=0,
                           kv_cache=None)
            x = x - (t - r) * u

        self.assertEqual(x.shape[1], 20)  # vc_chunk mel frames
        self.assertEqual(x.shape[2], 80)  # mel bins

        # Vocoder
        mel = x.transpose(1, 2)
        mel = (mel + 1) / 2
        wav_out = self.vocoder.decode(mel).squeeze()
        self.assertGreater(wav_out.numel(), 0)

        # Output should be valid audio range
        self.assertTrue(torch.all(wav_out >= -1.5))
        self.assertTrue(torch.all(wav_out <= 1.5))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
