#!/usr/bin/env python3
"""
Extract speaker embedding and prompt mel from a target voice audio file.

This is a standalone, one-time offline step. Once extracted, the VC inference
pipeline never needs to load WavLM again — it reads the saved .npy files directly.

Usage:
    python extract_spk.py --input target.wav --name "speaker_name"
    python extract_spk.py --input target.wav --name "speaker_name" --output-dir my_embeddings/
"""
import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import librosa
from librosa.filters import mel as librosa_mel_fn

# Ensure the runtime module is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "runtime"))


# ═══════════════════════════════════════════════════════════════════════
# Mel extraction (same as infer_ref.py / main.py)
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


# ═══════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_speaker_embedding(
    input_path: str,
    output_dir: str,
    sv_ckpt: str = "src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth",
):
    """
    Extract speaker embedding & prompt mel from a target audio file,
    save as .npy, and return memory stats.
    """
    from speaker_verification.verification import init_model as init_sv_model

    torch.set_num_threads(1)
    process = psutil.Process()
    mem_start = process.memory_info().rss / (1024 * 1024)

    # --- Load audio ---
    print(f"Loading audio: {input_path}")
    wav, sr = librosa.load(input_path, sr=16000)
    duration = len(wav) / sr
    print(f"  Duration: {duration:.1f}s, Samples: {len(wav)}")

    # --- Load WavLM ---
    print("Loading WavLM speaker verification model...")
    sv_model = init_sv_model('wavlm_large', sv_ckpt)
    sv_model.eval()
    mem_after_wavlm = process.memory_info().rss / (1024 * 1024)
    print(f"  Memory after loading WavLM: {mem_after_wavlm:.0f} MB")

    # --- Load Mel extractor ---
    mel_extract = MelSpectrogramFeatures(
        sample_rate=16000, n_fft=1024, win_size=640,
        hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True
    )

    # --- Extract ---
    print("Extracting speaker embedding & prompt mel...")
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)

    with torch.no_grad():
        spk_emb = sv_model(wav_tensor)              # (1, 256) — keep batch dim
        prompt_mel = mel_extract(wav_tensor)         # (1, 80, T)

    spk_emb = spk_emb                                 # (1, 256)
    prompt_mel = prompt_mel.transpose(1, 2)           # (1, 80, T) → (1, T, 80)

    print(f"  spk_emb shape: {spk_emb.shape}, dtype: {spk_emb.dtype}")
    print(f"  prompt_mel shape: {prompt_mel.shape}, dtype: {prompt_mel.dtype}")

    # --- Unload WavLM ---
    print("Unloading WavLM...")
    del sv_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_after_unload = process.memory_info().rss / (1024 * 1024)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    spk_path = os.path.join(output_dir, "spk_emb.npy")
    prompt_path = os.path.join(output_dir, "prompt_mel.npy")

    np.save(spk_path, spk_emb.cpu().numpy())
    np.save(prompt_path, prompt_mel.cpu().numpy())

    # Save metadata
    with open(os.path.join(output_dir, "source.txt"), "w") as f:
        f.write(f"source: {os.path.abspath(input_path)}\n")
        f.write(f"duration: {duration:.1f}s\n")
        f.write(f"sample_rate: {sr}\n")

    spk_size = os.path.getsize(spk_path)
    prompt_size = os.path.getsize(prompt_path)

    print(f"\n{'='*50}")
    print(f"Saved to: {output_dir}/")
    print(f"  spk_emb.npy     {spk_size:,} bytes ({spk_size/1024:.1f} KB)")
    print(f"  prompt_mel.npy  {prompt_size:,} bytes ({prompt_size/1024:.1f} KB)")
    print(f"  source.txt")
    print(f"\nMemory:")
    print(f"  Baseline:          {mem_start:.0f} MB")
    print(f"  With WavLM:        {mem_after_wavlm:.0f} MB")
    print(f"  After unloading:   {mem_after_unload:.0f} MB")
    print(f"  WavLM cost:        {mem_after_wavlm - mem_after_unload:.0f} MB")
    print(f"\n✅ Done. Now run main.py and select this embedding directory.")
    print(f"   WavLM is NOT needed at inference time.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embedding from target voice audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_spk.py --input target.wav --name "caixukun"
  python extract_spk.py --input target.wav --name "my_voice" --output-dir my_embeddings/
  python extract_spk.py --input target.wav --output-dir target_voice/embeddings/speaker1/
        """.strip(),
    )
    parser.add_argument("--input", required=True, help="Path to target speaker audio file (.wav, etc.)")
    parser.add_argument("--name", default=None, help="Speaker name (used as subdirectory name under embeddings/)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (overrides --name if specified)")
    parser.add_argument("--sv-ckpt", default="src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth",
                        help="Path to WavLM finetuned checkpoint")
    args = parser.parse_args()

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.name:
        output_dir = os.path.join("target_voice", "embeddings", args.name)
    else:
        # Derive name from filename
        name = os.path.splitext(os.path.basename(args.input))[0]
        output_dir = os.path.join("target_voice", "embeddings", name)

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.sv_ckpt):
        print(f"ERROR: WavLM checkpoint not found: {args.sv_ckpt}")
        print("Please download it from Google Drive and place it at:")
        print("  src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth")
        sys.exit(1)

    extract_speaker_embedding(args.input, output_dir, args.sv_ckpt)


if __name__ == "__main__":
    main()
