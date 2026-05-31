"""
Offline diagnostic test: compare original run_rt.py VCRunner vs GUI VCWorker.
Processes a real speech file through both pipelines and saves outputs.
Run: python temp/debug_conversion.py
"""
import sys
import os
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "runtime"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import torch
import librosa
import soundfile as sf

# Reuse the original VCRunner logic exactly
from speaker_verification.verification import init_model as init_sv_model
import torchaudio.compliance.kaldi as kaldi
from librosa.filters import mel as librosa_mel_fn


# Copy the original helpers EXACTLY
def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))

def _normalize(S, max_abs_value, min_db):
    return torch.clamp((2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value, -max_abs_value, max_abs_value)

class MelSpectrogramFeatures(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True):
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
            mel = librosa_mel_fn(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(self.win_size).to(dtype=y.dtype, device=y.device)
        spec = torch.stft(y, self.n_fft, hop_length=self.hop_length, win_length=self.win_size,
                        window=self.hann_window[wnsize_dtype_device],
                        center=self.center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = torch.matmul(self.mel_basis[fmax_dtype_device], spec)
        spec = _amp_to_db(spec, -115) - 20
        spec = _normalize(spec, 1, -115)
        return spec

def extract_fbanks(wav, sample_rate=16000, mel_bins=80, frame_length=25, frame_shift=12.5):
    wav = wav * (1 << 15)
    wav = torch.from_numpy(wav).unsqueeze(0)
    fbanks = kaldi.fbank(wav, frame_length=frame_length, frame_shift=frame_shift,
                         snip_edges=True, num_mel_bins=mel_bins, energy_floor=0.0,
                         dither=0.0, sample_frequency=sample_rate)
    fbanks = fbanks.unsqueeze(0)
    return fbanks


# ─── Original VCRunner inference_one_chunk (verbatim copy) ───────────

class OriginalVCRunner:
    def __init__(self, target_path, steps, sv_model, asr, vc, vocoder, mel_extract):
        self.sv_model = sv_model
        self.asr = asr
        self.vc = vc
        self.vocoder = vocoder
        self.mel_extract = mel_extract
        self.steps = steps
        if self.steps == 1:
            self.timesteps = torch.tensor([1.0, 0.0])
        elif self.steps == 2:
            self.timesteps = torch.tensor([1.0, 0.8, 0.0])
        else:
            self.timesteps = torch.linspace(1.0, 0.0, self.steps + 1)

        decoding_chunk_size = 5
        num_decoding_left_chunks = 2
        subsampling = 4
        context = 7
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        self.required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        self.CHUNK = 160 * stride
        self.vc_chunk = int(decoding_chunk_size * 4)
        self.vocoder_overlap = 3
        upsample_factor = 160
        self.vocoder_wav_overlap = (self.vocoder_overlap - 1) * upsample_factor
        self.down_linspace = torch.linspace(1, 0, steps=self.vocoder_wav_overlap, out=None).numpy()
        self.up_linspace = torch.linspace(0, 1, steps=self.vocoder_wav_overlap, out=None).numpy()

        wav, _ = librosa.load(target_path, sr=16000)
        wav = torch.from_numpy(wav).unsqueeze(0)
        spk_emb = self.sv_model(wav)
        self.vc_spk_emb = spk_emb
        prompt_mel = self.mel_extract(wav)
        prompt_mel = prompt_mel.transpose(1, 2)
        self.vc_prompt_mel = prompt_mel

    def init_cache(self):
        self.samples_cache_len = 720
        self.samples_cache = None
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

    def reset_cache(self):
        self.asr_offset = 20
        self.vc_offset = 120

    def inference_one_chunk(self, samples):
        with torch.no_grad():
            if self.samples_cache is None:
                samples = samples
            else:
                samples = np.concatenate((self.samples_cache, samples))
            self.samples_cache = samples[-self.samples_cache_len:]
            fbanks = extract_fbanks(samples, frame_shift=10).float()
            (encoder_output, self.att_cache, self.cnn_cache) = self.asr.forward_encoder_chunk(
                fbanks, self.asr_offset, self.required_cache_size, self.att_cache, self.cnn_cache)
            self.asr_offset += encoder_output.size(1)
            if self.encoder_output_cache is None:
                encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
            else:
                encoder_output = torch.cat([self.encoder_output_cache, encoder_output], dim=1)
            self.encoder_output_cache = encoder_output[:, -1:, :]
            encoder_output_upsample = encoder_output.transpose(1, 2)
            encoder_output_upsample = torch.nn.functional.interpolate(
                encoder_output_upsample, size=self.vc_chunk + 1, mode='linear', align_corners=True)
            encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
            encoder_output_upsample = encoder_output_upsample[:, 1:, :]

            x = torch.randn(1, encoder_output_upsample.shape[1], 80, device='cpu', dtype=encoder_output_upsample.dtype)

            for i in range(self.steps):
                t = self.timesteps[i]
                r = self.timesteps[i+1]
                t_tensor = torch.full((1,), t, device=x.device)
                r_tensor = torch.full((1,), r, device=x.device)
                u, tmp_kv_cache = self.vc(x, t_tensor, r_tensor, cache=self.vc_cache, cond=encoder_output_upsample,
                    spks=self.vc_spk_emb, prompts=self.vc_prompt_mel, offset=self.vc_offset, kv_cache=self.vc_kv_cache)
                x = x - (t - r) * u
            self.vc_kv_cache = tmp_kv_cache
            self.vc_offset += x.shape[1]
            self.vc_cache = x

            VC_KV_CACHE_MAX_LEN = 100
            if self.vc_offset > 40 and self.vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN:
                for i in range(len(self.vc_kv_cache)):
                    new_k = self.vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    new_v = self.vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                    self.vc_kv_cache[i] = (new_k, new_v)

            mel = x.transpose(1, 2)
            if self.vocoder_cache is not None:
                mel = torch.cat([self.vocoder_cache, mel], dim=-1)
            self.vocoder_cache = mel[:, :, -self.vocoder_overlap:]
            mel = (mel + 1) / 2
            wav = self.vocoder.decode(mel).squeeze()
            wav = wav.detach().cpu().numpy()

            if self.last_wav is not None:
                front_wav = wav[:self.vocoder_wav_overlap]
                smooth_front_wav = self.last_wav * self.down_linspace + front_wav * self.up_linspace
                new_wav = np.concatenate([smooth_front_wav, wav[self.vocoder_wav_overlap:-self.vocoder_wav_overlap]], axis=0)
            else:
                new_wav = wav[:-self.vocoder_wav_overlap]
            self.last_wav = wav[-self.vocoder_wav_overlap:]
            return new_wav


# ─── GUI VCWorker inference_one_chunk (verbatim copy) ───────────

class GUIWorkerClone:
    def __init__(self, target_path, steps, sv_model, asr, vc, vocoder, mel_extract):
        self.models = {
            'sv_model': sv_model, 'asr': asr, 'vc': vc, 'vocoder': vocoder,
            'mel_extract': mel_extract, 'steps': steps,
        }
        if steps == 1:
            self.models['timesteps'] = torch.tensor([1.0, 0.0])
        elif steps == 2:
            self.models['timesteps'] = torch.tensor([1.0, 0.8, 0.0])
        else:
            self.models['timesteps'] = torch.linspace(1.0, 0.0, steps + 1)

        wav, _ = librosa.load(target_path, sr=16000)
        wav_tensor = torch.from_numpy(wav).unsqueeze(0)
        self.models['vc_spk_emb'] = sv_model(wav_tensor)
        self.models['vc_prompt_mel'] = mel_extract(wav_tensor).transpose(1, 2)

    def init_cache(self):
        self.samples_cache_len = 720
        self.samples_cache = None
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

    def reset_cache(self):
        self.asr_offset = 20
        self.vc_offset = 120

    def inference_one_chunk(self, samples: np.ndarray) -> np.ndarray:
        m = self.models
        with torch.no_grad():
            if self.samples_cache is None:
                pass  # samples = samples
            else:
                samples = np.concatenate((self.samples_cache, samples))
            self.samples_cache = samples[-self.samples_cache_len:]

            fbanks = extract_fbanks(samples, frame_shift=10).float()
            required_cache_size = 5 * 2  # 10
            (encoder_output, self.att_cache, self.cnn_cache) = \
                m['asr'].forward_encoder_chunk(
                    fbanks, self.asr_offset, required_cache_size, self.att_cache, self.cnn_cache)

            self.asr_offset += encoder_output.size(1)
            if self.encoder_output_cache is None:
                encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
            else:
                encoder_output = torch.cat([self.encoder_output_cache, encoder_output], dim=1)
            self.encoder_output_cache = encoder_output[:, -1:, :]

            vc_chunk = int(5 * 4)  # 20
            encoder_output_upsample = encoder_output.transpose(1, 2)
            encoder_output_upsample = torch.nn.functional.interpolate(
                encoder_output_upsample, size=vc_chunk + 1, mode='linear', align_corners=True)
            encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
            encoder_output_upsample = encoder_output_upsample[:, 1:, :]

            x = torch.randn(1, encoder_output_upsample.shape[1], 80, device='cpu', dtype=encoder_output_upsample.dtype)

            for i in range(m['steps']):
                t = m['timesteps'][i]
                r = m['timesteps'][i + 1]
                t_tensor = torch.full((1,), t, device=x.device)
                r_tensor = torch.full((1,), r, device=x.device)
                u, tmp_kv_cache = m['vc'](x, t_tensor, r_tensor,
                    cache=self.vc_cache, cond=encoder_output_upsample,
                    spks=m['vc_spk_emb'], prompts=m['vc_prompt_mel'],
                    offset=self.vc_offset, kv_cache=self.vc_kv_cache)
                x = x - (t - r) * u

            self.vc_kv_cache = tmp_kv_cache
            self.vc_offset += x.shape[1]
            self.vc_cache = x

            VC_KV_CACHE_MAX_LEN = 100
            if self.vc_offset > 40 and self.vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN:
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
            wav = m['vocoder'].decode(mel).squeeze()
            wav = wav.detach().cpu().numpy()

            upsample_factor = 160
            vocoder_wav_overlap = (vocoder_overlap - 1) * upsample_factor
            down_linspace = np.linspace(1, 0, num=vocoder_wav_overlap)
            up_linspace = np.linspace(0, 1, num=vocoder_wav_overlap)

            if self.last_wav is not None:
                front_wav = wav[:vocoder_wav_overlap]
                smooth_front_wav = self.last_wav * down_linspace + front_wav * up_linspace
                new_wav = np.concatenate([
                    smooth_front_wav,
                    wav[vocoder_wav_overlap:-vocoder_wav_overlap]
                ], axis=0)
            else:
                new_wav = wav[:-vocoder_wav_overlap]
            self.last_wav = wav[-vocoder_wav_overlap:]

            return new_wav


# ─── Main test ──────────────────────────────────────────────────────

def main():
    torch.set_num_threads(1)

    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    sv_model = init_sv_model('wavlm_large', 'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth')
    sv_model.eval()
    asr = torch.jit.load('src/ckpt/fastu2++.pt')
    vc = torch.jit.load('src/ckpt/meanvc_200ms.pt')
    vocoder = torch.jit.load('src/ckpt/vocos.pt')
    mel_extract = MelSpectrogramFeatures(sample_rate=16000, n_fft=1024, win_size=640, hop_length=160, n_mels=80, fmin=0, fmax=8000, center=True)

    TARGET = "src/runtime/example/test.wav"
    SOURCE = "src/runtime/example/test.wav"  # use same file for both

    print(f"Target: {TARGET}")
    print(f"Source: {SOURCE}")

    # Create both runners
    print("\n--- Creating original VCRunner ---")
    orig = OriginalVCRunner(TARGET, steps=2, sv_model=sv_model, asr=asr, vc=vc, vocoder=vocoder, mel_extract=mel_extract)
    print("--- Creating GUI worker clone ---")
    gui = GUIWorkerClone(TARGET, steps=2, sv_model=sv_model, asr=asr, vc=vc, vocoder=vocoder, mel_extract=mel_extract)

    # Load source audio
    source_wav, _ = librosa.load(SOURCE, sr=16000)
    print(f"Source audio: {len(source_wav)} samples ({len(source_wav)/16000:.2f}s)")

    # Simulate streaming: feed chunks
    CHUNK_SAMPLES = 160 * 20  # 3200 — matches stride
    EXTRA = 720

    torch.manual_seed(42)
    np.random.seed(42)

    orig.init_cache()
    gui.init_cache()

    orig_wavs = []
    gui_wavs = []

    total_chunks = min(len(source_wav) // CHUNK_SAMPLES, 20)  # process up to 20 chunks

    print(f"\n{'='*60}")
    print(f"Processing {total_chunks} chunks...")
    print(f"{'='*60}")

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        chunk_data = source_wav[start:end].astype(np.float32)

        # Simulate what run_rt does: first chunk gets extra data
        if chunk_idx == 0:
            # Get extra data from source (or pad if source is short)
            if end + EXTRA <= len(source_wav):
                extra_data = source_wav[end:end + EXTRA].astype(np.float32)
            else:
                extra_data = np.zeros(EXTRA, dtype=np.float32)
            samples_orig = np.concatenate([chunk_data, extra_data])
            samples_gui = np.concatenate([chunk_data.copy(), extra_data.copy()])
        else:
            samples_orig = chunk_data.copy()
            samples_gui = chunk_data.copy()

        # Reset seed before each call so both get the same random x
        torch.manual_seed(42 + chunk_idx)
        out_orig = orig.inference_one_chunk(samples_orig)

        torch.manual_seed(42 + chunk_idx)
        out_gui = gui.inference_one_chunk(samples_gui)

        orig_wavs.append(out_orig)
        gui_wavs.append(out_gui)

        # Compare intermediate values for first chunk only (debug)
        if chunk_idx == 0:
            print(f"\n--- Chunk {chunk_idx} comparison ---")
            print(f"  Input samples length: orig={len(samples_orig)}, gui={len(samples_gui)}")
            print(f"  Output samples length: orig={len(out_orig)}, gui={len(out_gui)}")
            print(f"  Output abs diff max: {np.max(np.abs(out_orig - out_gui)):.6f}")

            # Compare internal state
            print(f"  asr_offset: orig={orig.asr_offset}, gui={gui.asr_offset}")
            print(f"  vc_offset: orig={orig.vc_offset}, gui={gui.vc_offset}")
            print(f"  vc_cache shape: orig={orig.vc_cache.shape if orig.vc_cache is not None else None}, gui={gui.vc_cache.shape if gui.vc_cache is not None else None}")

    # Concatenate outputs
    full_orig = np.concatenate(orig_wavs)
    full_gui = np.concatenate(gui_wavs)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"  Original output: {len(full_orig)} samples ({len(full_orig)/16000:.2f}s)")
    print(f"  GUI output:      {len(full_gui)} samples ({len(full_gui)/16000:.2f}s)")

    # Compare
    min_len = min(len(full_orig), len(full_gui))
    diff = np.abs(full_orig[:min_len] - full_gui[:min_len])
    print(f"  Max absolute difference: {np.max(diff):.6f}")
    print(f"  Mean absolute difference: {np.mean(diff):.6f}")
    print(f"  Correlation coefficient: {np.corrcoef(full_orig[:min_len], full_gui[:min_len])[0,1]:.6f}")

    # Check output characteristics
    print(f"  Original RMS: {np.sqrt(np.mean(full_orig**2)):.4f}")
    print(f"  GUI RMS:      {np.sqrt(np.mean(full_gui**2)):.4f}")

    if np.max(diff) < 1e-5:
        print("\n✅ EXACT MATCH — both pipelines produce identical output!")
    elif np.corrcoef(full_orig[:min_len], full_gui[:min_len])[0,1] > 0.999:
        print("\n✅ ESSENTIALLY IDENTICAL — correlation > 0.999")
    else:
        print("\n❌ OUTPUTS DIFFER — need to investigate")

    # Save outputs for listening tests
    os.makedirs("temp", exist_ok=True)
    sf.write("temp/debug_original.wav", full_orig.astype(np.float32), 16000)
    sf.write("temp/debug_gui.wav", full_gui.astype(np.float32), 16000)
    print(f"\nSaved debug outputs: temp/debug_original.wav, temp/debug_gui.wav")

    # Also save the source for reference
    sf.write("temp/debug_source.wav", source_wav[:len(full_orig)].astype(np.float32), 16000)
    print(f"Saved source reference: temp/debug_source.wav")


if __name__ == "__main__":
    main()
