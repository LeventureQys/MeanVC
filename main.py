#!/usr/bin/env python3
"""
MeanVC - Real-Time Voice Conversion GUI
Built with PyQt6, wrapping the MeanVC streaming inference pipeline.
"""

import sys
import os
import time
import json
import signal
import traceback
from pathlib import Path

# Ensure the runtime module is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "runtime"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gc
import numpy as np
import torch
import torch.nn as nn
import pyaudio
import librosa
from librosa.filters import mel as librosa_mel_fn
import torchaudio.compliance.kaldi as kaldi
import psutil

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog,
    QProgressBar, QSpinBox, QGridLayout, QSplitter, QFrame,
    QMessageBox, QSizePolicy, QTabWidget
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QMutex, QWaitCondition
)
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon


# ═══════════════════════════════════════════════════════════════════════
# Audio feature extraction utilities (from run_rt.py)
# ═══════════════════════════════════════════════════════════════════════

def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S, max_abs_value, min_db):
    return torch.clamp(
        (2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value,
        -max_abs_value, max_abs_value
    )


class MelSpectrogramFeatures(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, win_size=640,
                 hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_basis = {}
        self.hann_window = {}

    def forward(self, y):
        dtype_device = str(y.dtype) + '_' + str(y.device)
        fmax_dtype_device = str(self.fmax) + '_' + dtype_device
        wnsize_dtype_device = str(self.win_size) + '_' + dtype_device
        if fmax_dtype_device not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.sample_rate, n_fft=self.n_fft,
                                 n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
                dtype=y.dtype, device=y.device)
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(
                self.win_size).to(dtype=y.dtype, device=y.device)

        spec = torch.stft(
            y, self.n_fft, hop_length=self.hop_length, win_length=self.win_size,
            window=self.hann_window[wnsize_dtype_device],
            center=self.center, pad_mode='reflect', normalized=False,
            onesided=True, return_complex=False
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)
        spec = _amp_to_db(spec, -115) - 20
        spec = _normalize(spec, 1, -115)
        return spec


def extract_fbanks(wav, sample_rate=16000, mel_bins=80,
                   frame_length=25, frame_shift=12.5):
    wav = wav * (1 << 15)
    wav = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(
        wav, frame_length=frame_length, frame_shift=frame_shift,
        snip_edges=True, num_mel_bins=mel_bins, energy_floor=0.0,
        dither=0.0, sample_frequency=sample_rate,
    )
    fbanks = fbanks.unsqueeze(0)
    return fbanks


# ═══════════════════════════════════════════════════════════════════════
# Model Loader Thread
# ═══════════════════════════════════════════════════════════════════════

class EmbeddingExtractor(QThread):
    """Extracts speaker embedding + prompt mel from target audio (uses WavLM).

    This is a ONE-TIME extraction thread. Once the .npy files are saved to disk,
    WavLM is no longer needed — the VC pipeline reads the .npy files directly.
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float, float)   # output_dir, mem_peak_mb, mem_after_mb
    error = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir

    def run(self):
        try:
            from speaker_verification.verification import init_model as init_sv_model

            sv_ckpt = 'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth'
            if not os.path.exists(sv_ckpt):
                self.error.emit(
                    f"WavLM checkpoint not found:\n  {sv_ckpt}\n\n"
                    "Please download it manually from Google Drive."
                )
                return

            torch.set_num_threads(1)
            process = psutil.Process()
            mem_start = process.memory_info().rss / (1024 * 1024)

            # --- Load WavLM ---
            self.progress.emit("Loading WavLM speaker verification model...")
            sv_model = init_sv_model('wavlm_large', sv_ckpt)
            sv_model.eval()
            mem_with_wavlm = process.memory_info().rss / (1024 * 1024)

            # --- Load audio ---
            self.progress.emit("Loading target audio...")
            wav, _ = librosa.load(self.input_path, sr=16000)
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)

            # --- Extract ---
            self.progress.emit("Extracting speaker embedding & prompt mel...")
            mel_extract = MelSpectrogramFeatures(
                sample_rate=16000, n_fft=1024, win_size=640,
                hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True
            )
            with torch.no_grad():
                spk_emb = sv_model(wav_tensor)                     # (1, 256) — keep batch dim for TorchScript
                prompt_mel = mel_extract(wav_tensor).transpose(1, 2)   # (1, 80, T) → (1, T, 80)

            # --- Unload WavLM ---
            self.progress.emit("Unloading WavLM (saving memory)...")
            del sv_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mem_after = process.memory_info().rss / (1024 * 1024)

            # --- Save ---
            os.makedirs(self.output_dir, exist_ok=True)
            np.save(os.path.join(self.output_dir, "spk_emb.npy"),
                    spk_emb.cpu().numpy())
            np.save(os.path.join(self.output_dir, "prompt_mel.npy"),
                    prompt_mel.cpu().numpy())
            with open(os.path.join(self.output_dir, "source.txt"), "w") as f:
                f.write(f"source: {os.path.abspath(self.input_path)}\n")
                f.write(f"duration: {len(wav)/16000:.1f}s\n")

            self.progress.emit(
                f"✅ Saved to {self.output_dir}/  |  "
                f"WavLM peak: {mem_with_wavlm:.0f} MB → released: {mem_after:.0f} MB"
            )
            self.finished.emit(self.output_dir, mem_with_wavlm, mem_after)

        except Exception:
            self.error.emit(f"Extraction failed:\n{traceback.format_exc()}")


class ModelLoader(QThread):
    """Loads VC models (ASR, DiT, Vocoder) + reads pre-extracted speaker embedding.

    WavLM is NOT loaded here — the embedding must be pre-extracted using
    extract_spk.py or the EmbeddingExtractor tab.
    """
    progress = pyqtSignal(str)
    finished_loading = pyqtSignal(dict)
    error = pyqtSignal(str)

    REQUIRED_FILES = [
        ('src/ckpt/fastu2++.pt', 'ASR model'),
        ('src/ckpt/meanvc_200ms.pt', 'Voice conversion model'),
        ('src/ckpt/vocos.pt', 'Vocoder model'),
    ]

    @staticmethod
    def check_required_files():
        missing = []
        for path, desc in ModelLoader.REQUIRED_FILES:
            if not os.path.exists(path):
                missing.append(f"  • {desc}: {path}")
        return len(missing) == 0, missing

    @staticmethod
    def list_embeddings(base_dir: str = None):
        """Return list of (name, dir_path) for all saved embedding directories."""
        if base_dir is None:
            base_dir = os.path.join(PROJECT_ROOT, "target_voice", "embeddings")
        if not os.path.isdir(base_dir):
            return []
        results = []
        for name in sorted(os.listdir(base_dir)):
            d = os.path.join(base_dir, name)
            if os.path.isdir(d):
                spk = os.path.join(d, "spk_emb.npy")
                prompt = os.path.join(d, "prompt_mel.npy")
                if os.path.exists(spk) and os.path.exists(prompt):
                    results.append((name, d))
        return results

    def __init__(self, embedding_dir: str, steps: int = 2):
        super().__init__()
        self.embedding_dir = embedding_dir
        self.steps = steps

    def run(self):
        try:
            ok, missing = self.check_required_files()
            if not ok:
                self.error.emit(
                    "Missing required model files:\n" +
                    "\n".join(missing) +
                    "\n\nPlease run: python download_ckpt.py"
                )
                return

            torch.set_num_threads(1)
            process = psutil.Process()

            # --- Read pre-extracted speaker embedding ---
            spk_path = os.path.join(self.embedding_dir, "spk_emb.npy")
            prompt_path = os.path.join(self.embedding_dir, "prompt_mel.npy")
            if not os.path.exists(spk_path) or not os.path.exists(prompt_path):
                self.error.emit(
                    f"Incomplete embedding directory:\n"
                    f"  Expected: {spk_path}\n"
                    f"  Expected: {prompt_path}\n\n"
                    f"Please run extract_spk.py first, or use the Extract tab."
                )
                return

            # --- Baseline: RSS before loading any model ---
            rss_baseline = process.memory_info().rss / (1024 * 1024)

            self.progress.emit("Loading pre-extracted speaker embedding...")
            vc_spk_emb = torch.from_numpy(np.load(spk_path))
            vc_prompt_mel = torch.from_numpy(np.load(prompt_path))
            rss_embed = process.memory_info().rss / (1024 * 1024)
            # Embedding .npy tensors are tiny (~KB), so delta ≈ 0

            # --- Timesteps ---
            if self.steps == 1:
                timesteps = torch.tensor([1.0, 0.0])
            elif self.steps == 2:
                timesteps = torch.tensor([1.0, 0.8, 0.0])
            else:
                timesteps = torch.linspace(1.0, 0.0, self.steps + 1)

            # --- ASR Model (FastU2++ 21.7M params, TorchScript) ---
            self.progress.emit("Loading ASR model (21.7M)...")
            asr = torch.jit.load('src/ckpt/fastu2++.pt')
            rss_asr = process.memory_info().rss / (1024 * 1024)

            # --- VC Model (DiT 14.1M params, TorchScript) ---
            self.progress.emit("Loading VC model (14.1M)...")
            vc = torch.jit.load("src/ckpt/meanvc_200ms.pt")
            rss_dit = process.memory_info().rss / (1024 * 1024)

            # --- Vocoder (Vocos 8.3M params, TorchScript) ---
            self.progress.emit("Loading Vocoder (8.3M)...")
            vocoder = torch.jit.load('src/ckpt/vocos.pt')
            rss_final = process.memory_info().rss / (1024 * 1024)

            # --- Compute per-component deltas ---
            asr_delta   = rss_asr   - rss_embed   # ASR 增量
            dit_delta   = rss_dit   - rss_asr     # DiT 增量
            vocos_delta = rss_final - rss_dit     # Vocos 增量
            net_core    = rss_final - rss_baseline # 纯核心管线净增

            self.progress.emit(
                f"  📊 RSS baseline (Qt+Python): {rss_baseline:.0f} MB"
            )
            self.progress.emit(
                f"  ➕ ASR   (21.7M): +{asr_delta:.0f} MB  →  {rss_asr:.0f} MB"
            )
            self.progress.emit(
                f"  ➕ DiT   (14.1M): +{dit_delta:.0f} MB  →  {rss_dit:.0f} MB"
            )
            self.progress.emit(
                f"  ➕ Vocos ( 8.3M): +{vocos_delta:.0f} MB  →  {rss_final:.0f} MB"
            )
            memory_report = (
                f"  💾 Core operators NET: {net_core:.0f} MB  (Total RSS: {rss_final:.0f} MB)"
            )
            self.progress.emit(memory_report)

            models = {
                'asr': asr,
                'vc': vc,
                'vocoder': vocoder,
                'vc_spk_emb': vc_spk_emb,
                'vc_prompt_mel': vc_prompt_mel,
                'timesteps': timesteps,
                'steps': self.steps,
                'memory_report': memory_report,
                'mem_core_mb': rss_final,
                'mem_net_core_mb': net_core,
                'rss_baseline_mb': rss_baseline,    # Qt+Python overhead (subtract from RSS)
                'embedding_dir': self.embedding_dir,
            }
            self.progress.emit("Models loaded successfully!")
            self.finished_loading.emit(models)

        except Exception:
            self.error.emit(f"Failed to load models:\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════
# Voice Conversion Worker Thread
# ═══════════════════════════════════════════════════════════════════════

class VCWorker(QThread):
    """Runs the real-time voice conversion loop in a background thread."""

    status_update = pyqtSignal(str)
    metrics_update = pyqtSignal(float, float, float)  # chunk_ms, total_latency_ms, memory_mb
    error_signal = pyqtSignal(str)
    processing_started = pyqtSignal()
    processing_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.models = None
        self.input_device_id = None
        self.output_device_id = None
        self.in_stream = None
        self.out_stream = None
        self.pyaudio_instance = None
        self._running = False
        self._pause = False
        self._mutex = QMutex()

        # State variables
        self.samples_cache = None
        self.samples_cache_len = 720
        self.att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
        self.cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device='cpu')
        self.asr_offset = 0
        self.encoder_output_cache = None
        self.vc_offset = 0
        self.vc_cache = None
        self.vc_kv_cache = None
        self.vocoder_cache = None
        self.last_wav = None
        self.need_extra_data = True
        self.chunk_counter = 0

    def set_models(self, models: dict):
        self.models = models

    def set_devices(self, input_id: int, output_id: int):
        self.input_device_id = input_id
        self.output_device_id = output_id

    def set_streams(self, in_stream, out_stream, pyaudio_instance):
        """Pre-opened PyAudio streams — must be created in main thread on macOS."""
        self.in_stream = in_stream
        self.out_stream = out_stream
        self.pyaudio_instance = pyaudio_instance

    def init_cache(self):
        self.samples_cache_len = 720
        self.samples_cache = None
        self.att_cache = torch.zeros((0, 0, 0, 0), device='cpu')
        self.cnn_cache = torch.zeros((0, 0, 0, 0), device='cpu')
        self.asr_offset = 0
        self.encoder_output_cache = None
        self.vc_offset = 0
        self.vc_cache = None
        self.vc_kv_cache = None
        self.vocoder_cache = None
        self.last_wav = None
        self.need_extra_data = True
        self.chunk_counter = 0

    def reset_cache(self):
        self.asr_offset = 20
        self.vc_offset = 120

    def inference_one_chunk(self, samples: np.ndarray) -> np.ndarray:
        m = self.models
        with torch.no_grad():
            if self.samples_cache is None:
                pass
            else:
                samples = np.concatenate((self.samples_cache, samples))
            self.samples_cache = samples[-self.samples_cache_len:]

            fbanks = extract_fbanks(samples, frame_shift=10).float()

            required_cache_size = 5 * 2
            (encoder_output, self.att_cache, self.cnn_cache) = \
                m['asr'].forward_encoder_chunk(
                    fbanks, self.asr_offset, required_cache_size,
                    self.att_cache, self.cnn_cache
                )

            self.asr_offset += encoder_output.size(1)
            if self.encoder_output_cache is None:
                encoder_output = torch.cat(
                    [encoder_output[:, 0:1, :], encoder_output], dim=1)
            else:
                encoder_output = torch.cat(
                    [self.encoder_output_cache, encoder_output], dim=1)
            self.encoder_output_cache = encoder_output[:, -1:, :]

            vc_chunk = int(5 * 4)
            encoder_output_upsample = encoder_output.transpose(1, 2)
            encoder_output_upsample = torch.nn.functional.interpolate(
                encoder_output_upsample, size=vc_chunk + 1,
                mode='linear', align_corners=True
            )
            encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
            encoder_output_upsample = encoder_output_upsample[:, 1:, :]

            x = torch.randn(
                1, encoder_output_upsample.shape[1], 80,
                device='cpu', dtype=encoder_output_upsample.dtype
            )

            for i in range(m['steps']):
                t = m['timesteps'][i]
                r = m['timesteps'][i + 1]
                t_tensor = torch.full((1,), t, device=x.device)
                r_tensor = torch.full((1,), r, device=x.device)
                u, tmp_kv_cache = m['vc'](
                    x, t_tensor, r_tensor,
                    cache=self.vc_cache,
                    cond=encoder_output_upsample,
                    spks=m['vc_spk_emb'],
                    prompts=m['vc_prompt_mel'],
                    offset=self.vc_offset,
                    kv_cache=self.vc_kv_cache,
                )
                x = x - (t - r) * u

            self.vc_kv_cache = tmp_kv_cache
            self.vc_offset += x.shape[1]
            self.vc_cache = x

            VC_KV_CACHE_MAX_LEN = 100
            if (self.vc_offset > 40 and
                    self.vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN):
                for i in range(len(self.vc_kv_cache)):
                    new_k = self.vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    new_v = self.vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    self.vc_kv_cache[i] = (new_k, new_v)

            mel = x.transpose(1, 2)

            vocoder_overlap = 3
            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -vocoder_overlap:]
            mel = (mel + 1) / 2
            wav_raw = m['vocoder'].decode(mel).squeeze()
            wav_raw = wav_raw.detach().cpu().numpy()

            upsample_factor = 160
            vocoder_wav_overlap = (vocoder_overlap - 1) * upsample_factor
            down_linspace = np.linspace(1, 0, num=vocoder_wav_overlap)
            up_linspace = np.linspace(0, 1, num=vocoder_wav_overlap)

            if self.last_wav is not None:
                front_wav = wav_raw[:vocoder_wav_overlap]
                smooth_front_wav = (
                    self.last_wav * down_linspace + front_wav * up_linspace
                )
                new_wav = np.concatenate([
                    smooth_front_wav,
                    wav_raw[vocoder_wav_overlap:-vocoder_wav_overlap]
                ], axis=0)
            else:
                new_wav = wav_raw[:-vocoder_wav_overlap]
            # CRITICAL: save raw vocoder output tail, NOT the trimmed/cross-faded output
            self.last_wav = wav_raw[-vocoder_wav_overlap:]

            return new_wav

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()

    def pause(self):
        self._mutex.lock()
        self._pause = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._pause = False
        self._mutex.unlock()

    def run(self):
        """Main VC processing loop with comprehensive diagnostic logging."""
        import soundfile as sf

        # Prepare log directory
        log_dir = os.path.join(PROJECT_ROOT, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "vc_worker.log")

        def log(msg: str, level: str = "INFO"):
            ts = time.strftime("%H:%M:%S")
            line = f"[{ts}] [{level}] {msg}"
            print(line)
            with open(log_path, "a") as f:
                f.write(line + "\n")

        try:
            m = self.models
            torch.set_num_threads(1)

            log(f"{'='*60}")
            log(f"VCWorker started")
            log(f"  input_device_id: {self.input_device_id}")
            log(f"  output_device_id: {self.output_device_id}")
            log(f"  steps: {m['steps']}")
            log(f"  timesteps: {m['timesteps'].tolist()}")
            log(f"  vc_spk_emb: shape={m['vc_spk_emb'].shape}, dtype={m['vc_spk_emb'].dtype}, "
                f"min={m['vc_spk_emb'].min():.4f}, max={m['vc_spk_emb'].max():.4f}, "
                f"mean={m['vc_spk_emb'].mean():.4f}")
            log(f"  vc_prompt_mel: shape={m['vc_prompt_mel'].shape}, dtype={m['vc_prompt_mel'].dtype}, "
                f"min={m['vc_prompt_mel'].min():.4f}, max={m['vc_prompt_mel'].max():.4f}, "
                f"mean={m['vc_prompt_mel'].mean():.4f}")
            log(f"  ASR model type: {type(m['asr'])}")
            log(f"  VC model type: {type(m['vc'])}")
            log(f"  Vocoder type: {type(m['vocoder'])}")

            # Verify stream info
            try:
                log(f"  in_stream is_active: {self.in_stream.is_active()}")
                log(f"  out_stream is_active: {self.out_stream.is_active()}")
            except Exception as e:
                log(f"  stream status check failed: {e}", "WARN")

            decoding_chunk_size = 5
            subsampling = 4
            stride = subsampling * decoding_chunk_size
            CHUNK = 160 * stride  # 3200
            log(f"  CHUNK (samples per read): {CHUNK}")
            log(f"  decoding_chunk_size: {decoding_chunk_size}")

            # Warmup + save a sample for diagnostics
            self.status_update.emit("Warming up...")
            log("--- Warmup (5 chunks) ---")
            warmup_audio = []
            for i_warm in range(5):
                data = self.in_stream.read(CHUNK, exception_on_overflow=False)
                raw = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
                warmup_audio.append(raw)
                self.out_stream.write(raw.tobytes())
                log(f"  warmup[{i_warm}]: read {len(data)} bytes → {len(raw)} samples, "
                    f"min={raw.min():.4f}, max={raw.max():.4f}, rms={np.sqrt(np.mean(raw**2)):.4f}")
            time.sleep(0.1)

            # Save warmup audio
            warmup_all = np.concatenate(warmup_audio)
            sf.write(os.path.join(log_dir, "warmup_input.wav"), warmup_all, 16000)
            log(f"Saved warmup audio: {len(warmup_all)} samples ({len(warmup_all)/16000:.2f}s)")

            self.init_cache()
            self.status_update.emit("Converting...")
            self.processing_started.emit()

            process = psutil.Process()
            chunk_idx = 0

            log(f"{'='*60}")
            log(f"Starting conversion loop")

            while self._running:
                if self._pause:
                    time.sleep(0.01)
                    continue

                if chunk_idx % 50 == 0 and chunk_idx != 0:
                    log(f"Periodic reset at chunk {chunk_idx}")
                    self.reset_cache()

                # ── Read microphone ──
                data = self.in_stream.read(CHUNK, exception_on_overflow=False)
                n_bytes = len(data)
                samples_in = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)

                if self.need_extra_data:
                    extra_data = self.in_stream.read(720, exception_on_overflow=False)
                    extra_samples = np.frombuffer(extra_data, dtype=np.int16).astype(np.float32) / (1 << 15)
                    samples_in = np.concatenate([samples_in, extra_samples])
                    self.need_extra_data = False

                n_samples_in = len(samples_in)

                # ── Save input audio (first 5 chunks) ──
                if chunk_idx < 5:
                    sf.write(os.path.join(log_dir, f"chunk_{chunk_idx:03d}_input.wav"),
                             samples_in.astype(np.float32), 16000)

                # ── Inference (INLINE — exact copy of debug_mic_loop logic) ──
                t_start = time.perf_counter()

                with torch.no_grad():
                    # Concatenate with sample cache
                    _samples = samples_in
                    if self.samples_cache is None:
                        pass  # _samples = _samples
                    else:
                        _samples = np.concatenate((self.samples_cache, _samples))
                    self.samples_cache = _samples[-self.samples_cache_len:]

                    # Extract fbanks
                    fbanks = extract_fbanks(_samples, frame_shift=10).float()

                    # ASR encoder chunk
                    req_cache = 5 * 2  # 10
                    (enc_out, self.att_cache, self.cnn_cache) = \
                        m['asr'].forward_encoder_chunk(
                            fbanks, self.asr_offset, req_cache,
                            self.att_cache, self.cnn_cache)
                    self.asr_offset += enc_out.size(1)

                    # Prepare encoder output
                    if self.encoder_output_cache is None:
                        enc_out = torch.cat([enc_out[:, 0:1, :], enc_out], dim=1)
                    else:
                        enc_out = torch.cat([self.encoder_output_cache, enc_out], dim=1)
                    self.encoder_output_cache = enc_out[:, -1:, :]

                    # Upsample BN features to mel frame rate
                    vc_chunk = int(5 * 4)  # 20
                    enc_up = enc_out.transpose(1, 2)
                    enc_up = torch.nn.functional.interpolate(
                        enc_up, size=vc_chunk + 1, mode='linear', align_corners=True)
                    enc_up = enc_up.transpose(1, 2)
                    enc_up = enc_up[:, 1:, :]

                    # Initialize noise and run CFM ODE solver
                    x = torch.randn(1, enc_up.shape[1], 80, device='cpu', dtype=enc_up.dtype)

                    for i_step in range(m['steps']):
                        t = m['timesteps'][i_step]
                        r = m['timesteps'][i_step + 1]
                        t_tensor = torch.full((1,), t, device=x.device)
                        r_tensor = torch.full((1,), r, device=x.device)
                        u, tmp_kv_cache = m['vc'](
                            x, t_tensor, r_tensor,
                            cache=self.vc_cache,
                            cond=enc_up,
                            spks=m['vc_spk_emb'],
                            prompts=m['vc_prompt_mel'],
                            offset=self.vc_offset,
                            kv_cache=self.vc_kv_cache,
                        )
                        x = x - (t - r) * u

                    self.vc_kv_cache = tmp_kv_cache
                    self.vc_offset += x.shape[1]
                    self.vc_cache = x

                    # KV cache management
                    VC_KV_CACHE_MAX_LEN = 100
                    if (self.vc_offset > 40 and
                            self.vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN):
                        for i in range(len(self.vc_kv_cache)):
                            new_k = self.vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                            new_v = self.vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                            self.vc_kv_cache[i] = (new_k, new_v)

                    # Vocoder
                    mel = x.transpose(1, 2)
                    voc_overlap = 3
                    if self.vocoder_cache is not None:
                        mel = torch.cat([self.vocoder_cache, mel], dim=-1)
                    self.vocoder_cache = mel[:, :, -voc_overlap:]
                    mel_norm = (mel + 1) / 2
                    wav_raw = m['vocoder'].decode(mel_norm).squeeze()
                    wav_raw = wav_raw.detach().cpu().numpy()

                    # Overlap-add smoothing
                    upsample_factor = 160
                    voc_wav_overlap = (voc_overlap - 1) * upsample_factor
                    down_linspace = np.linspace(1, 0, num=voc_wav_overlap)
                    up_linspace = np.linspace(0, 1, num=voc_wav_overlap)

                    if self.last_wav is not None:
                        front_wav = wav_raw[:voc_wav_overlap]
                        smooth_front = self.last_wav * down_linspace + front_wav * up_linspace
                        vc_wav = np.concatenate([
                            smooth_front,
                            wav_raw[voc_wav_overlap:-voc_wav_overlap]
                        ], axis=0)
                    else:
                        vc_wav = wav_raw[:-voc_wav_overlap]
                    # CRITICAL: save raw vocoder output tail, NOT the trimmed/cross-faded output
                    self.last_wav = wav_raw[-voc_wav_overlap:]

                chunk_time = (time.perf_counter() - t_start) * 1000.0

                # ── Playback ──
                self.out_stream.write(vc_wav.astype(np.float32).tobytes())

                # ── Save output audio (first 5 chunks) ──
                if chunk_idx < 5:
                    sf.write(os.path.join(log_dir, f"chunk_{chunk_idx:03d}_output.wav"),
                             vc_wav.astype(np.float32), 16000)

                # ── Metrics ──
                output_duration = len(vc_wav) / 16000 * 1000.0
                total_latency = chunk_time + output_duration
                memory_mb = (process.memory_info().rss / (1024 * 1024)
                             - m.get('rss_baseline_mb', 0))
                self.metrics_update.emit(chunk_time, total_latency, memory_mb)

                # ── Detailed log (first 10 chunks, then every 30) ──
                if chunk_idx < 10 or chunk_idx % 30 == 0:
                    log(
                        f"chunk[{chunk_idx:03d}]: "
                        f"read_bytes={n_bytes} in_samples={n_samples_in} "
                        f"out_samples={len(vc_wav)} "
                        f"fbanks={list(fbanks.shape)} "
                        f"enc_out={list(enc_out.shape)} "
                        f"enc_up={list(enc_up.shape)} "
                        f"x_mean={x.mean():.4f} x_std={x.std():.4f} "
                        f"x_range=[{x.min():.3f},{x.max():.3f}] "
                        f"vc_offset={self.vc_offset} "
                        f"wav_rms={np.sqrt(np.mean(vc_wav**2)):.4f} "
                        f"wav_peak={np.abs(vc_wav).max():.3f} "
                        f"time={chunk_time:.1f}ms"
                    )

                chunk_idx += 1

        except Exception as e:
            log(f"FATAL ERROR: {traceback.format_exc()}", "ERROR")
            self.error_signal.emit(f"VC processing error:\n{traceback.format_exc()}")
        finally:
            log(f"Cleaning up — processed {chunk_idx} chunks total")

            # Save final chunk counter for diagnostics
            try:
                with open(os.path.join(log_dir, "summary.txt"), "w") as sf_sum:
                    sf_sum.write(f"total_chunks={chunk_idx}\n")
                    sf_sum.write(f"final_vc_offset={self.vc_offset}\n")
                    sf_sum.write(f"final_asr_offset={self.asr_offset}\n")
            except:
                pass

            if self.in_stream:
                try: self.in_stream.stop_stream()
                except: pass
                try: self.in_stream.close()
                except: pass
            if self.out_stream:
                try: self.out_stream.stop_stream()
                except: pass
                try: self.out_stream.close()
                except: pass
            if self.pyaudio_instance:
                try: self.pyaudio_instance.terminate()
                except: pass
            log(f"Cleanup complete")
            self.status_update.emit("Stopped")
            self.processing_stopped.emit()


# ═══════════════════════════════════════════════════════════════════════
# Audio Device Scanner
# ═══════════════════════════════════════════════════════════════════════

class DeviceScanner(QThread):
    """Scans audio devices in background."""
    devices_ready = pyqtSignal(list, list, int, int)  # inputs, outputs, default_in, default_out

    def run(self):
        try:
            import pyaudio as pa
            p = pa.PyAudio()
            info = p.get_host_api_info_by_index(0)
            num = info.get('deviceCount')

            input_devs = []
            output_devs = []

            for i in range(num):
                dev = p.get_device_info_by_host_api_device_index(0, i)
                if dev.get('maxInputChannels') > 0:
                    input_devs.append({
                        'id': i,
                        'name': dev.get('name'),
                        'channels': dev.get('maxInputChannels'),
                    })
                if dev.get('maxOutputChannels') > 0:
                    output_devs.append({
                        'id': i,
                        'name': dev.get('name'),
                        'channels': dev.get('maxOutputChannels'),
                    })

            default_in = p.get_default_input_device_info()['index']
            default_out = p.get_default_output_device_info()['index']
            p.terminate()

            self.devices_ready.emit(input_devs, output_devs, default_in, default_out)
        except Exception:
            self.devices_ready.emit([], [], -1, -1)


# ═══════════════════════════════════════════════════════════════════════
# Performance Metrics Widget
# ═══════════════════════════════════════════════════════════════════════

class MetricWidget(QWidget):
    """Displays a single performance metric with label and value."""

    def __init__(self, title: str, unit: str, color: str = "#ffffff"):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: #888; font-size: 11px;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.value_label = QLabel(f"-- {unit}")
        self.value_label.setStyleSheet(
            f"color: {color}; font-size: 18px; font-weight: bold;"
        )
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.value_label.setMinimumWidth(110)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)
        self.setMinimumHeight(60)

    def set_value(self, value: float, fmt: str = ".1f"):
        self.value_label.setText(f"{value:{fmt}} {self.unit}")

    def set_text(self, text: str):
        self.value_label.setText(text)


class MetricsPanel(QGroupBox):
    """Panel showing real-time performance metrics."""

    def __init__(self):
        super().__init__("📊 Performance Metrics")
        self.setStyleSheet("""
            QGroupBox {
                color: #aaa;
                font-weight: bold;
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)
        self.setMinimumHeight(160)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setContentsMargins(12, 8, 12, 8)

        self.chunk_time = MetricWidget("Chunk Processing", "ms", "#00d4ff")
        self.total_latency = MetricWidget("Total Latency", "ms", "#ffd700")
        self.memory = MetricWidget("Framework Memory", "MB", "#7ec8e3")
        self.realtime_factor = MetricWidget("Real-Time Factor", "x", "#00ff88")
        self.chunk_rate = MetricWidget("Chunk Rate", "/s", "#ff6b6b")

        grid.addWidget(self.chunk_time, 0, 0)
        grid.addWidget(self.total_latency, 0, 1)
        grid.addWidget(self.memory, 0, 2)
        grid.addWidget(self.realtime_factor, 1, 0)
        grid.addWidget(self.chunk_rate, 1, 1)

        # Equal column width distribution
        for col in range(3):
            grid.setColumnStretch(col, 1)
        for row in range(2):
            grid.setRowStretch(row, 1)

        self.setLayout(grid)

        # Rolling stats
        self._chunk_times = []
        self._chunk_rate_times = []
        self._last_update = time.perf_counter()

    def update_metrics(self, chunk_time_ms: float, total_latency_ms: float,
                       memory_mb: float):
        now = time.perf_counter()
        self._chunk_times.append(chunk_time_ms)
        self._chunk_rate_times.append(now)

        # Keep last 50 samples
        if len(self._chunk_times) > 50:
            self._chunk_times = self._chunk_times[-50:]
        if len(self._chunk_rate_times) > 50:
            self._chunk_rate_times = self._chunk_rate_times[-50:]

        avg_chunk = np.mean(self._chunk_times) if self._chunk_times else chunk_time_ms

        # Real-time factor: processing time / audio duration
        # audio duration per chunk: CHUNK samples / 16000 Hz * 1000 ms
        audio_dur_ms = (720 + 2560) / 16000 * 1000  # approximate
        rtf = avg_chunk / audio_dur_ms if audio_dur_ms > 0 else 0

        # Chunk rate
        if len(self._chunk_rate_times) >= 2:
            elapsed = self._chunk_rate_times[-1] - self._chunk_rate_times[0]
            rate = len(self._chunk_rate_times) / elapsed if elapsed > 0 else 0
        else:
            rate = 0

        self.chunk_time.set_text(f"{avg_chunk:.1f} ms")
        self.total_latency.set_text(f"{total_latency_ms:.1f} ms")
        self.memory.set_text(f"{memory_mb:.0f} MB")
        self.realtime_factor.set_text(f"{rtf:.2f} x")
        self.chunk_rate.set_text(f"{rate:.1f} /s")

    def reset(self):
        self._chunk_times.clear()
        self._chunk_rate_times.clear()
        self.chunk_time.set_text("-- ms")
        self.total_latency.set_text("-- ms")
        self.memory.set_text("-- MB")
        self.realtime_factor.set_text("-- x")
        self.chunk_rate.set_text("-- /s")


# ═══════════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════════

class MeanVCApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MeanVC — Real-Time Voice Conversion")
        self.setMinimumSize(640, 560)
        self.resize(720, 600)

        # Application state
        self.models: dict | None = None
        self.vc_worker: VCWorker | None = None
        self.model_loader: ModelLoader | None = None
        self.embed_extractor: EmbeddingExtractor | None = None
        self.device_scanner: DeviceScanner | None = None
        self.input_devices = []
        self.output_devices = []
        self.target_path = ""
        self.embedding_dirs = []  # [(name, dir_path), ...]

        self._setup_ui()
        self._apply_theme()
        self._refresh_devices()
        self._refresh_embeddings()

    # ── UI Setup ──────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(16, 12, 16, 12)

        # --- Title ---
        title = QLabel("🎙️ MeanVC")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #fff;")
        root.addWidget(title)

        subtitle = QLabel(
            "Lightweight & Streaming Zero-Shot Voice Conversion"
        )
        subtitle.setStyleSheet("font-size: 13px; color: #888; margin-bottom: 6px;")
        root.addWidget(subtitle)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #333;
                border-radius: 4px;
                background: #12121a;
            }
            QTabBar::tab {
                background: #1e1e2e;
                color: #888;
                border: 1px solid #333;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #2a2a3a;
                color: #fff;
                border-bottom: 2px solid #00d4ff;
            }
        """)

        # === Tab 1: Extract Speaker Embedding ===
        extract_tab = QWidget()
        extract_layout = QVBoxLayout(extract_tab)
        extract_layout.setSpacing(12)
        extract_layout.setContentsMargins(16, 16, 16, 16)

        # Description
        extract_desc = QLabel(
            "Extract speaker embedding from a target voice audio file.\n"
            "This step uses WavLM (~1.2GB) ONCE, then saves a tiny .npy file.\n"
            "After extraction, WavLM is no longer needed for voice conversion."
        )
        extract_desc.setStyleSheet("color: #888; font-size: 12px;")
        extract_desc.setWordWrap(True)
        extract_layout.addWidget(extract_desc)

        # Audio file picker
        extract_file_group = QGroupBox("Target Audio File")
        extract_file_group.setStyleSheet(self._group_style())
        efg_layout = QHBoxLayout()
        self.extract_file_label = QLabel("No file selected")
        self.extract_file_label.setStyleSheet(
            "color: #888; background: #1e1e2e; border: 1px solid #333; "
            "border-radius: 4px; padding: 6px 10px;"
        )
        self.extract_file_label.setWordWrap(True)
        efg_layout.addWidget(self.extract_file_label, 1)
        browse_extract_btn = QPushButton("Browse...")
        browse_extract_btn.setStyleSheet(self._btn_style())
        browse_extract_btn.clicked.connect(self._browse_extract_file)
        efg_layout.addWidget(browse_extract_btn)
        extract_file_group.setLayout(efg_layout)
        extract_layout.addWidget(extract_file_group)

        # Speaker name
        name_group = QGroupBox("Speaker Name")
        name_group.setStyleSheet(self._group_style())
        ng_layout = QHBoxLayout()
        self.speaker_name_input = QLineEdit()
        self.speaker_name_input.setPlaceholderText("e.g. caixukun, zhenhuan, my_voice...")
        self.speaker_name_input.setStyleSheet(
            "QLineEdit { background: #1e1e2e; border: 1px solid #333; "
            "border-radius: 4px; padding: 6px 10px; color: #fff; }"
        )
        ng_layout.addWidget(QLabel("Name:"))
        ng_layout.addWidget(self.speaker_name_input, 1)
        ng_layout.addWidget(QLabel("→ target_voice/embeddings/<name>/"))
        name_group.setLayout(ng_layout)
        extract_layout.addWidget(name_group)

        # Extract button + progress
        extract_btn_row = QHBoxLayout()
        self.extract_btn = QPushButton("🔧 Extract Speaker Embedding")
        self.extract_btn.setStyleSheet(self._btn_style(accent=True, green=True))
        self.extract_btn.clicked.connect(self._start_extraction)
        self.extract_btn.setMinimumHeight(40)
        extract_btn_row.addWidget(self.extract_btn)

        self.extract_progress = QProgressBar()
        self.extract_progress.setRange(0, 0)
        self.extract_progress.setVisible(False)
        self.extract_progress.setMaximumHeight(6)
        self.extract_progress.setStyleSheet(
            "QProgressBar { background: #1e1e2e; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #00d4ff; border-radius: 3px; }"
        )
        extract_btn_row.addWidget(self.extract_progress)
        extract_layout.addLayout(extract_btn_row)

        # Extract status
        self.extract_status = QLabel("")
        self.extract_status.setStyleSheet("color: #888; font-size: 12px;")
        self.extract_status.setWordWrap(True)
        extract_layout.addWidget(self.extract_status)

        extract_layout.addStretch()
        self.tab_widget.addTab(extract_tab, "① Extract Speaker")

        # === Tab 2: Voice Conversion ===
        convert_tab = QWidget()
        convert_layout = QVBoxLayout(convert_tab)
        convert_layout.setSpacing(12)
        convert_layout.setContentsMargins(16, 16, 16, 16)

        # Embedding selector
        embed_group = QGroupBox("Target Speaker (Pre-Extracted Embedding)")
        embed_group.setStyleSheet(self._group_style())
        eg_layout = QHBoxLayout()
        self.embed_combo = QComboBox()
        self.embed_combo.setStyleSheet(self._combo_style())
        self.embed_combo.setMinimumWidth(250)
        eg_layout.addWidget(QLabel("Speaker:"))
        eg_layout.addWidget(self.embed_combo, 1)
        refresh_embed_btn = QPushButton("🔄 Refresh")
        refresh_embed_btn.setStyleSheet(self._btn_style())
        refresh_embed_btn.clicked.connect(self._refresh_embeddings)
        eg_layout.addWidget(refresh_embed_btn)
        embed_group.setLayout(eg_layout)
        convert_layout.addWidget(embed_group)

        # Steps selector
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Denoising steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 8)
        self.steps_spin.setValue(2)
        self.steps_spin.setToolTip(
            "1 = fastest, lower quality\n2 = default (recommended)\n"
            "3+ = higher quality, slower\n"
        )
        self.steps_spin.setStyleSheet(
            "QSpinBox { background: #1e1e2e; border: 1px solid #333; "
            "border-radius: 4px; padding: 4px 8px; color: #fff; }"
        )
        step_row.addWidget(self.steps_spin)
        step_row.addStretch()
        convert_layout.addLayout(step_row)

        # Audio Devices
        device_group = QGroupBox("Audio Devices")
        device_group.setStyleSheet(self._group_style())
        dg_layout = QGridLayout()
        dg_layout.setSpacing(8)
        dg_layout.addWidget(QLabel("Input (Microphone):"), 0, 0)
        self.input_combo = QComboBox()
        self.input_combo.setStyleSheet(self._combo_style())
        dg_layout.addWidget(self.input_combo, 0, 1)
        dg_layout.addWidget(QLabel("Output (Speaker):"), 1, 0)
        self.output_combo = QComboBox()
        self.output_combo.setStyleSheet(self._combo_style())
        dg_layout.addWidget(self.output_combo, 1, 1)
        refresh_dev_btn = QPushButton("🔄 Refresh Devices")
        refresh_dev_btn.setStyleSheet(self._btn_style())
        refresh_dev_btn.clicked.connect(self._refresh_devices)
        dg_layout.addWidget(refresh_dev_btn, 0, 2, 2, 1)
        device_group.setLayout(dg_layout)
        convert_layout.addWidget(device_group)

        # --- Status Bar ---
        status_row = QHBoxLayout()
        self.status_icon = QLabel("⚪")
        self.status_icon.setStyleSheet("font-size: 16px;")
        self.status_label = QLabel("Ready — Please use headphones to avoid feedback!")
        self.status_label.setStyleSheet("color: #ffd700; font-size: 13px;")
        status_row.addWidget(self.status_icon)
        status_row.addWidget(self.status_label, 1)

        self.load_progress = QProgressBar()
        self.load_progress.setRange(0, 0)
        self.load_progress.setVisible(False)
        self.load_progress.setMaximumHeight(6)
        self.load_progress.setStyleSheet(
            "QProgressBar { background: #1e1e2e; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #00d4ff; border-radius: 3px; }"
        )
        status_row.addWidget(self.load_progress)
        convert_layout.addLayout(status_row)

        # --- Action Buttons ---
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.load_btn = QPushButton("📦 Load Models")
        self.load_btn.setStyleSheet(self._btn_style(accent=True))
        self.load_btn.clicked.connect(self._load_models)
        self.load_btn.setMinimumHeight(40)
        btn_row.addWidget(self.load_btn)

        self.start_btn = QPushButton("▶ Start Conversion")
        self.start_btn.setStyleSheet(self._btn_style(accent=True, green=True))
        self.start_btn.clicked.connect(self._start_conversion)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(self._btn_style(accent=True, red=True))
        self.stop_btn.clicked.connect(self._stop_conversion)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.stop_btn)

        convert_layout.addLayout(btn_row)

        # --- Metrics Panel ---
        self.metrics = MetricsPanel()
        convert_layout.addWidget(self.metrics)

        # --- Memory Breakdown Label ---
        self.mem_label = QLabel("")
        self.mem_label.setStyleSheet(
            "color: #7ec8e3; font-size: 11px; background: #1a1a2e; "
            "border: 1px solid #333; border-radius: 4px; padding: 4px 10px;"
        )
        self.mem_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.mem_label.setVisible(False)
        convert_layout.addWidget(self.mem_label)

        self.tab_widget.addTab(convert_tab, "② Voice Conversion")

        root.addWidget(self.tab_widget)

        # --- Footer ---
        footer = QLabel("MeanVC • ASLP Lab • NPU")
        footer.setStyleSheet("color: #555; font-size: 11px;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(footer)

    # ── Theme & Styles ────────────────────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #12121a; }
            QWidget { color: #ddd; font-size: 13px; }
            QLabel { background: transparent; }
        """)

    def _group_style(self) -> str:
        return """
            QGroupBox {
                color: #aaa;
                font-weight: bold;
                border: 1px solid #333;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """

    def _btn_style(self, accent: bool = False, green: bool = False,
                   red: bool = False) -> str:
        if green:
            bg = "#0d7a3e"
            bg_hover = "#0f8e4a"
        elif red:
            bg = "#8b1a1a"
            bg_hover = "#a02020"
        elif accent:
            bg = "#1a5276"
            bg_hover = "#1f6090"
        else:
            bg = "#2a2a3a"
            bg_hover = "#3a3a4a"

        return f"""
            QPushButton {{
                background: {bg};
                color: #fff;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: {bg_hover}; }}
            QPushButton:disabled {{
                background: #1a1a22;
                color: #555;
                border-color: #2a2a2a;
            }}
        """

    def _combo_style(self) -> str:
        return """
            QComboBox {
                background: #1e1e2e;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 6px 10px;
                color: #fff;
                min-width: 200px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1e1e2e;
                border: 1px solid #444;
                color: #fff;
                selection-background-color: #1a5276;
            }
        """

    # ── Device Management ─────────────────────────────────────────────

    def _refresh_devices(self):
        """Scan audio devices in background."""
        self.input_combo.clear()
        self.output_combo.clear()
        self.input_combo.addItem("Scanning...", -1)
        self.output_combo.addItem("Scanning...", -1)

        self.device_scanner = DeviceScanner()
        self.device_scanner.devices_ready.connect(self._on_devices_ready)
        self.device_scanner.start()

    def _on_devices_ready(self, inputs: list, outputs: list,
                          default_in: int, default_out: int):
        self.input_devices = inputs
        self.output_devices = outputs

        self.input_combo.clear()
        for i, dev in enumerate(inputs):
            label = f"{dev['name']}  [{dev['channels']}ch]"
            self.input_combo.addItem(label, dev['id'])
            if dev['id'] == default_in:
                self.input_combo.setCurrentIndex(i)

        self.output_combo.clear()
        for i, dev in enumerate(outputs):
            label = f"{dev['name']}  [{dev['channels']}ch]"
            self.output_combo.addItem(label, dev['id'])
            if dev['id'] == default_out:
                self.output_combo.setCurrentIndex(i)

        if not inputs:
            self.input_combo.addItem("No input devices found", -1)
        if not outputs:
            self.output_combo.addItem("No output devices found", -1)

    # ── Extraction Tab ────────────────────────────────────────────────

    def _browse_extract_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Speaker Audio",
            str(PROJECT_ROOT / "target_voice"),
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*)",
        )
        if path:
            self.target_path = path
            display = path
            if len(display) > 60:
                display = "..." + display[-57:]
            self.extract_file_label.setText(display)
            self.extract_file_label.setStyleSheet(
                "color: #00d4ff; background: #1e1e2e; border: 1px solid #333; "
                "border-radius: 4px; padding: 6px 10px;"
            )
            # Auto-suggest name from filename
            base = os.path.splitext(os.path.basename(path))[0]
            if not self.speaker_name_input.text():
                self.speaker_name_input.setText(base)
            self.extract_btn.setEnabled(True)

    def _start_extraction(self):
        if not self.target_path:
            return

        name = self.speaker_name_input.text().strip()
        if not name:
            name = os.path.splitext(os.path.basename(self.target_path))[0]
            self.speaker_name_input.setText(name)

        output_dir = os.path.join(
            PROJECT_ROOT, "target_voice", "embeddings", name
        )
        if os.path.exists(output_dir):
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"Embedding for '{name}' already exists.\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.extract_btn.setEnabled(False)
        self.extract_progress.setVisible(True)
        self.extract_status.setStyleSheet("color: #ffd700; font-size: 12px;")
        self.extract_status.setText("Starting extraction...")

        self.embed_extractor = EmbeddingExtractor(self.target_path, output_dir)
        self.embed_extractor.progress.connect(self._on_extract_progress)
        self.embed_extractor.finished.connect(self._on_extract_finished)
        self.embed_extractor.error.connect(self._on_extract_error)
        self.embed_extractor.start()

    def _on_extract_progress(self, msg: str):
        self.extract_status.setText(msg)

    def _on_extract_finished(self, output_dir: str, mem_peak: float, mem_after: float):
        self.extract_btn.setEnabled(True)
        self.extract_progress.setVisible(False)
        self.extract_status.setStyleSheet("color: #00ff88; font-size: 12px;")
        self.extract_status.setText(
            f"✅ Saved to: {output_dir}/\n"
            f"   WavLM peak: {mem_peak:.0f} MB → released, now: {mem_after:.0f} MB\n"
            f"   Switch to 'Voice Conversion' tab to use this speaker."
        )
        self._refresh_embeddings()
        # Auto-select the newly extracted embedding in the convert tab
        for i in range(self.embed_combo.count()):
            if self.embed_combo.itemData(i) == output_dir:
                self.embed_combo.setCurrentIndex(i)
                break

    def _on_extract_error(self, msg: str):
        self.extract_btn.setEnabled(True)
        self.extract_progress.setVisible(False)
        self.extract_status.setStyleSheet("color: #ff4444; font-size: 12px;")
        self.extract_status.setText(f"❌ {msg}")
        QMessageBox.critical(self, "Extraction Error", msg)

    # ── Embedding Management ──────────────────────────────────────────

    def _refresh_embeddings(self):
        """Scan target_voice/embeddings/ and populate the combo box."""
        self.embedding_dirs = ModelLoader.list_embeddings()
        self.embed_combo.clear()
        if self.embedding_dirs:
            for name, dir_path in self.embedding_dirs:
                # Parse source.txt for metadata
                src_file = os.path.join(dir_path, "source.txt")
                info_parts = [dir_path]
                duration_str = ""
                if os.path.exists(src_file):
                    with open(src_file) as f:
                        lines = f.read().strip().split("\n")
                        for line in lines:
                            if line.startswith("duration:"):
                                duration_str = line.split(":")[1].strip() + "s"
                            elif line.startswith("source:"):
                                src_name = os.path.basename(line.split(":", 1)[1].strip())
                                info_parts.append(f"source: {src_name}")
                # Display: name + duration; full info in per-item tooltip
                label = f"🎤 {name}"
                if duration_str:
                    label += f"  ({duration_str})"
                idx = self.embed_combo.count()
                self.embed_combo.addItem(label, dir_path)
                self.embed_combo.setItemData(
                    idx, "\n".join(info_parts), Qt.ItemDataRole.ToolTipRole
                )
            self.load_btn.setEnabled(True)
        else:
            self.embed_combo.addItem("No embeddings found — extract one first!", "")
            self.load_btn.setEnabled(False)

    # ── Model Loading ─────────────────────────────────────────────────

    def _load_models(self):
        embed_dir = self.embed_combo.currentData()
        if not embed_dir or not os.path.isdir(embed_dir):
            QMessageBox.warning(self, "No Embedding",
                                "Please select a pre-extracted speaker embedding.\n\n"
                                "Use the 'Extract Speaker' tab first.")
            return

        input_id = self.input_combo.currentData()
        output_id = self.output_combo.currentData()
        if input_id is None or input_id < 0:
            QMessageBox.warning(self, "No Input Device",
                                "Please select a valid input device.")
            return
        if output_id is None or output_id < 0:
            QMessageBox.warning(self, "No Output Device",
                                "Please select a valid output device.")
            return

        self._set_ui_loading(True)
        self._set_status("Loading models...", "#ffd700")

        self.model_loader = ModelLoader(embed_dir, self.steps_spin.value())
        self.model_loader.progress.connect(self._on_loader_progress)
        self.model_loader.finished_loading.connect(self._on_models_loaded)
        self.model_loader.error.connect(self._on_loader_error)
        self.model_loader.start()

    def _on_models_loaded(self, models: dict):
        self.models = models
        self._set_ui_loading(False)
        self._set_status("Models loaded — Ready to convert", "#00ff88")
        self.start_btn.setEnabled(True)
        self.load_progress.setVisible(False)

        # Show memory
        net_core = models.get('mem_net_core_mb', 0)
        total = models.get('mem_core_mb', 0)
        embed_name = os.path.basename(models.get('embedding_dir', ''))
        self.mem_label.setText(
            f"💾 Core operators (NET): {net_core:.0f} MB  |  "
            f"Total RSS: {total:.0f} MB (incl. Qt+Python)  |  "
            f"Speaker: {embed_name}  |  "
            f"ASR(21.7M) + DiT(14.1M) + Vocos(8.3M)"
        )
        self.mem_label.setVisible(True)

    def _set_ui_loading(self, loading: bool):
        self.load_btn.setEnabled(not loading)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.load_progress.setVisible(loading)

    def _on_loader_progress(self, msg: str):
        self._set_status(msg, "#ffd700")

    def _on_loader_error(self, msg: str):
        self._set_ui_loading(False)
        self._set_status("Failed to load models", "#ff4444")
        self.load_progress.setVisible(False)
        QMessageBox.critical(self, "Model Loading Error", msg)

    # ── Voice Conversion Control ──────────────────────────────────────

    def _start_conversion(self):
        if not self.models:
            QMessageBox.warning(self, "Models Not Loaded",
                                "Please load the models first.")
            return

        input_id = self.input_combo.currentData()
        output_id = self.output_combo.currentData()
        if input_id is None or input_id < 0 or output_id is None or output_id < 0:
            QMessageBox.warning(self, "Invalid Device",
                                "Please select valid audio devices.")
            return

        # Open PyAudio in MAIN THREAD (macOS CoreAudio requires main-thread RunLoop)
        try:
            import pyaudio as pa
            CHUNK = 160 * 20  # 3200 samples
            pya_instance = pa.PyAudio()

            # Log device info
            log_dir = os.path.join(PROJECT_ROOT, "log")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "device_info.txt"), "w") as df:
                df.write(f"input_device_id={input_id}\n")
                df.write(f"output_device_id={output_id}\n")
                try:
                    di = pya_instance.get_device_info_by_index(input_id)
                    df.write(f"input_name={di['name']}\n")
                    df.write(f"input_max_input_ch={di['maxInputChannels']}\n")
                    df.write(f"input_default_sr={di['defaultSampleRate']}\n")
                except:
                    pass
                try:
                    do = pya_instance.get_device_info_by_index(output_id)
                    df.write(f"output_name={do['name']}\n")
                    df.write(f"output_max_output_ch={do['maxOutputChannels']}\n")
                    df.write(f"output_default_sr={do['defaultSampleRate']}\n")
                except:
                    pass
                df.write(f"CHUNK={CHUNK}\n")
                pya_instance.terminate()

            pya_instance = pa.PyAudio()
            in_stream = pya_instance.open(
                format=pa.paInt16, channels=1, rate=16000,
                input=True, input_device_index=input_id,
                frames_per_buffer=CHUNK,
            )
            out_stream = pya_instance.open(
                format=pa.paFloat32, channels=1, rate=16000,
                output=True, output_device_index=output_id,
            )
        except Exception as e:
            QMessageBox.critical(self, "Audio Error",
                                 f"Failed to open audio device:\n{e}")
            return

        self.vc_worker = VCWorker()
        self.vc_worker.set_models(self.models)
        self.vc_worker.set_devices(input_id, output_id)
        self.vc_worker.set_streams(in_stream, out_stream, pya_instance)

        self.vc_worker.status_update.connect(self._set_status_gui)
        self.vc_worker.metrics_update.connect(self._on_metrics_update)
        self.vc_worker.error_signal.connect(self._on_vc_error)
        self.vc_worker.processing_started.connect(self._on_processing_started)
        self.vc_worker.processing_stopped.connect(self._on_processing_stopped)

        self.vc_worker._running = True
        self.vc_worker.start()

    def _stop_conversion(self):
        if self.vc_worker and self.vc_worker.isRunning():
            self._set_status("Stopping...", "#ffd700")
            self.vc_worker.stop()
            # Give it a moment to clean up
            QTimer.singleShot(2000, lambda: self._check_vc_stopped())

    def _check_vc_stopped(self):
        if self.vc_worker and not self.vc_worker.isRunning():
            self._on_processing_stopped()

    def _on_processing_started(self):
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.load_btn.setEnabled(False)
        self._set_status("🟢 Converting...", "#00ff88")
        self.metrics.reset()

    def _on_processing_stopped(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.load_btn.setEnabled(True)
        self._set_status("Stopped — Ready", "#888")
        self.metrics.reset()

    def _on_metrics_update(self, chunk_ms: float, total_ms: float, memory_mb: float):
        self.metrics.update_metrics(chunk_ms, total_ms, memory_mb)

    def _on_vc_error(self, msg: str):
        self._set_status("Error during conversion", "#ff4444")
        self._on_processing_stopped()
        QMessageBox.critical(self, "VC Error", msg)

    # ── Helpers ───────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = "#888"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 13px;")

    def _set_status_gui(self, text: str):
        """Called from VC worker thread, determines color from text."""
        if "error" in text.lower() or "fail" in text.lower():
            color = "#ff4444"
        elif "convert" in text.lower():
            color = "#00ff88"
        elif "load" in text.lower() or "warm" in text.lower():
            color = "#ffd700"
        elif "stop" in text.lower():
            color = "#888"
        else:
            color = "#aaa"
        self._set_status(text, color)

    # ── Cleanup ───────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Ensure proper cleanup on window close."""
        if self.vc_worker and self.vc_worker.isRunning():
            self.vc_worker.stop()
            self.vc_worker.wait(3000)
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.terminate()
            self.model_loader.wait(3000)
        if self.embed_extractor and self.embed_extractor.isRunning():
            self.embed_extractor.terminate()
            self.embed_extractor.wait(3000)
        event.accept()


# ═══════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # Change to project root
    os.chdir(PROJECT_ROOT)

    app = QApplication(sys.argv)
    app.setApplicationName("MeanVC")
    app.setOrganizationName("ASLP-Lab")

    window = MeanVCApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
