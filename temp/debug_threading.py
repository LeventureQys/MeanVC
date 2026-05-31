"""
Test: does loading models in a QThread vs main thread produce different inference output?
Run: python temp/debug_threading.py
"""
import sys, os, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "runtime"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import torch
import librosa

from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication
from speaker_verification.verification import init_model as init_sv_model
from main import MelSpectrogramFeatures, extract_fbanks

# ─── Helper: run one chunk inference and return output stats ──────

def run_one_inference(asr, vc, vocoder, vc_spk_emb, vc_prompt_mel, steps, timesteps):
    """Run a single chunk of inference and return (output_wav, stats_dict)."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Simulate audio chunk (3920 samples)
    samples = np.random.randn(3920).astype(np.float32) * 0.01

    samples_cache_len = 720
    decoding_chunk_size = 5
    required_cache_size = decoding_chunk_size * 2
    vc_chunk_size = int(decoding_chunk_size * 4)

    fbanks = extract_fbanks(samples, frame_shift=10).float()

    att_cache = torch.zeros((0, 0, 0, 0), device='cpu')
    cnn_cache = torch.zeros((0, 0, 0, 0), device='cpu')
    (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
        fbanks, 0, required_cache_size, att_cache, cnn_cache)

    encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
    encoder_up = encoder_output.transpose(1, 2)
    encoder_up = torch.nn.functional.interpolate(
        encoder_up, size=vc_chunk_size + 1, mode='linear', align_corners=True)
    encoder_up = encoder_up.transpose(1, 2)[:, 1:, :]

    x = torch.randn(1, encoder_up.shape[1], 80, dtype=encoder_up.dtype)

    for i in range(steps):
        t, r = timesteps[i], timesteps[i+1]
        t_t = torch.full((1,), t, device=x.device)
        r_t = torch.full((1,), r, device=x.device)
        u, _ = vc(x, t_t, r_t, cache=None, cond=encoder_up, spks=vc_spk_emb,
                  prompts=vc_prompt_mel, offset=0, kv_cache=None)
        x = x - (t - r) * u

    mel = x.transpose(1, 2)
    mel = (mel + 1) / 2
    wav_out = vocoder.decode(mel).squeeze()
    wav_out = wav_out.detach().cpu().numpy()

    return {
        'wav': wav_out,
        'rms': float(np.sqrt(np.mean(wav_out**2))),
        'max': float(np.max(np.abs(wav_out))),
        'mean': float(np.mean(wav_out)),
        'vc_output_mean': float(x.mean()),
        'vc_output_std': float(x.std()),
    }


# ─── Threaded loader (simulates ModelLoader) ─────────────────────

class ThreadedLoader(QThread):
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, target_path, steps):
        super().__init__()
        self.target_path = target_path
        self.steps = steps

    def run(self):
        try:
            torch.set_num_threads(1)
            sv_model = init_sv_model('wavlm_large', 'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth')
            sv_model.eval()
            asr = torch.jit.load('src/ckpt/fastu2++.pt')
            vc = torch.jit.load('src/ckpt/meanvc_200ms.pt')
            vocoder = torch.jit.load('src/ckpt/vocos.pt')
            mel_extract = MelSpectrogramFeatures()

            if self.steps == 1:
                timesteps = torch.tensor([1.0, 0.0])
            elif self.steps == 2:
                timesteps = torch.tensor([1.0, 0.8, 0.0])
            else:
                timesteps = torch.linspace(1.0, 0.0, self.steps + 1)

            wav, _ = librosa.load(self.target_path, sr=16000)
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)
            vc_spk_emb = sv_model(wav_tensor)
            vc_prompt_mel = mel_extract(wav_tensor).transpose(1, 2)

            self.result.emit({
                'asr': asr, 'vc': vc, 'vocoder': vocoder,
                'vc_spk_emb': vc_spk_emb, 'vc_prompt_mel': vc_prompt_mel,
                'steps': self.steps, 'timesteps': timesteps,
            })
        except Exception as e:
            self.error.emit(str(e))


# ─── Main test ───────────────────────────────────────────────────

def main():
    app = QCoreApplication(sys.argv)

    TARGET = "src/runtime/example/test.wav"

    # 1. Load models in MAIN thread (like debug_mic_loop)
    print("=== Loading in MAIN THREAD ===")
    torch.set_num_threads(1)
    sv_model_m = init_sv_model('wavlm_large', 'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth')
    sv_model_m.eval()
    asr_m = torch.jit.load('src/ckpt/fastu2++.pt')
    vc_m = torch.jit.load('src/ckpt/meanvc_200ms.pt')
    vocoder_m = torch.jit.load('src/ckpt/vocos.pt')
    mel_m = MelSpectrogramFeatures()

    wav_m, _ = librosa.load(TARGET, sr=16000)
    wav_t_m = torch.from_numpy(wav_m).unsqueeze(0)
    vc_spk_emb_m = sv_model_m(wav_t_m)
    vc_prompt_mel_m = mel_m(wav_t_m).transpose(1, 2)
    timesteps_m = torch.tensor([1.0, 0.8, 0.0])

    # 2. Load models in QThread (like GUI)
    print("=== Loading in QTHREAD ===")
    loader = ThreadedLoader(TARGET, steps=2)
    threaded_models = {}

    def on_result(models):
        threaded_models.update(models)

    def on_error(err):
        print(f"ERROR: {err}")

    loader.result.connect(on_result)
    loader.error.connect(on_error)
    loader.start()

    # Process events while waiting for the thread
    print("  Waiting for thread...")
    for _ in range(600):
        app.processEvents()
        if threaded_models:
            break
        time.sleep(0.1)

    if not threaded_models:
        print("FAILED TO LOAD IN THREAD! (timed out)")
        return

    print(f"Main-thread models: vc_spk_emb shape={vc_spk_emb_m.shape}, mean={vc_spk_emb_m.mean():.4f}")
    print(f"Threaded models:   vc_spk_emb shape={threaded_models['vc_spk_emb'].shape}, mean={threaded_models['vc_spk_emb'].mean():.4f}")

    # 3. Compare speaker embeddings
    emb_diff = (vc_spk_emb_m - threaded_models['vc_spk_emb']).abs().max().item()
    print(f"\nSpeaker embedding max difference: {emb_diff:.10f}")

    # 4. Compare prompt mels
    prompt_diff = (vc_prompt_mel_m - threaded_models['vc_prompt_mel']).abs().max().item()
    print(f"Prompt mel max difference: {prompt_diff:.10f}")

    # 5. Run inference with BOTH sets of models
    print(f"\n=== Running inference ===")

    # Main thread models
    print("\n--- Main-thread model inference ---")
    r1 = run_one_inference(asr_m, vc_m, vocoder_m, vc_spk_emb_m, vc_prompt_mel_m, 2, timesteps_m)
    print(f"  VC output: mean={r1['vc_output_mean']:.6f}, std={r1['vc_output_std']:.6f}")
    print(f"  Wav: rms={r1['rms']:.6f}, max={r1['max']:.6f}")

    # Threaded models
    print("\n--- Threaded model inference ---")
    r2 = run_one_inference(
        threaded_models['asr'], threaded_models['vc'], threaded_models['vocoder'],
        threaded_models['vc_spk_emb'], threaded_models['vc_prompt_mel'],
        2, threaded_models['timesteps'])
    print(f"  VC output: mean={r2['vc_output_mean']:.6f}, std={r2['vc_output_std']:.6f}")
    print(f"  Wav: rms={r2['rms']:.6f}, max={r2['max']:.6f}")

    # Compare
    wav_diff = np.abs(r1['wav'] - r2['wav']).max()
    print(f"\n=== COMPARISON ===")
    print(f"  Wav max diff: {wav_diff:.10f}")
    if wav_diff < 1e-5:
        print("  ✅ IDENTICAL — threading is NOT the issue")
    else:
        print(f"  ❌ DIFFER — threading causes divergence! (max diff = {wav_diff:.6f})")
        print(f"  Main RMS={r1['rms']:.6f}, Threaded RMS={r2['rms']:.6f}")
        print(f"  VC output mean: main={r1['vc_output_mean']:.6f}, threaded={r2['vc_output_mean']:.6f}")
        print(f"  VC output std:  main={r1['vc_output_std']:.6f}, threaded={r2['vc_output_std']:.6f}")

    app.quit()


if __name__ == "__main__":
    main()
