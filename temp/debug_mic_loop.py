"""
Practical mic → VC → file test. Records 5s from mic, processes through VCWorker,
saves output WAV and diagnostic info. Use headphones to avoid feedback!
Run: python temp/test_mic_loop.py
"""
import sys, os, time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "runtime"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import torch
import pyaudio
import librosa
import soundfile as sf

from speaker_verification.verification import init_model as init_sv_model
from main import MelSpectrogramFeatures

print("=" * 60)
print("Mic → VC Pipeline Test")
print("=" * 60)

# 1. List devices
p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
num = info.get('deviceCount')
print("\n=== Audio Devices ===")
input_devs, output_devs = [], []
for i in range(num):
    dev = p.get_device_info_by_host_api_device_index(0, i)
    if dev.get('maxInputChannels') > 0:
        input_devs.append((i, dev['name']))
        print(f"  IN  #{i}: {dev['name']}")
    if dev.get('maxOutputChannels') > 0:
        output_devs.append((i, dev['name']))
        print(f"  OUT #{i}: {dev['name']}")

default_in = p.get_default_input_device_info()['index']
default_out = p.get_default_output_device_info()['index']
print(f"\nDefault IN:  #{default_in}")
print(f"Default OUT: #{default_out}")

in_id = int(input(f"Select input device [{default_in}]: ") or default_in)
out_id = int(input(f"Select output device [{default_out}]: ") or default_out)
p.terminate()

# 2. Load models
print("\n--- Loading models ---")
torch.set_num_threads(1)

sv_model = init_sv_model('wavlm_large', 'src/runtime/speaker_verification/ckpt/wavlm_large_finetune.pth')
sv_model.eval()
asr = torch.jit.load('src/ckpt/fastu2++.pt')
vc_model = torch.jit.load('src/ckpt/meanvc_200ms.pt')
vocoder = torch.jit.load('src/ckpt/vocos.pt')
mel_extract = MelSpectrogramFeatures()

TARGET = "src/runtime/example/test.wav"
wav, _ = librosa.load(TARGET, sr=16000)
wav_tensor = torch.from_numpy(wav).unsqueeze(0)
vc_spk_emb = sv_model(wav_tensor)
vc_prompt_mel = mel_extract(wav_tensor).transpose(1, 2)

steps = 2
timesteps = torch.tensor([1.0, 0.8, 0.0])

# 3. Initialize state (matching VCRunner.init_cache)
samples_cache = None
samples_cache_len = 720
att_cache = torch.zeros((0, 0, 0, 0), device='cpu')
cnn_cache = torch.zeros((0, 0, 0, 0), device='cpu')
asr_offset = 0
encoder_output_cache = None
vc_offset = 0
vc_cache = None
vc_kv_cache = None
vocoder_cache = None
last_wav = None
need_extra_data = True

decoding_chunk_size = 5
required_cache_size = decoding_chunk_size * 2  # 10
vc_chunk_size = int(decoding_chunk_size * 4)  # 20
CHUNK = 160 * decoding_chunk_size * 4  # 3200

vocoder_overlap = 3
vocoder_wav_overlap = (vocoder_overlap - 1) * 160
down_linspace = torch.linspace(1, 0, steps=vocoder_wav_overlap, out=None).numpy()
up_linspace = torch.linspace(0, 1, steps=vocoder_wav_overlap, out=None).numpy()

print("Models loaded. Starting mic capture...")

# 4. Open streams
p = pyaudio.PyAudio()
in_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                   input=True, input_device_index=in_id,
                   frames_per_buffer=CHUNK)
out_stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000,
                    output=True, output_device_index=out_id)

# 5. Warmup
print("Warming up (raw passthrough)...")
for i in range(5):
    data = in_stream.read(CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)
    out_stream.write(samples.tobytes())
    print(f"  Warmup {i+1}: max={np.max(np.abs(samples)):.3f}, rms={np.sqrt(np.mean(samples**2)):.3f}")
time.sleep(0.1)

# 6. Process N chunks
NUM_CHUNKS = 25  # ~5 seconds
print(f"\nProcessing {NUM_CHUNKS} chunks ({NUM_CHUNKS * CHUNK / 16000:.1f}s of audio)...")
print("⚠️  PLEASE WEAR HEADPHONES to avoid feedback!")
print("🎤 Speak now...")

from main import extract_fbanks

output_wavs = []
input_snapshots = []
chunk_times = []

for chunk_idx in range(NUM_CHUNKS):
    t0 = time.perf_counter()

    # Read mic
    data = in_stream.read(CHUNK, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (1 << 15)

    if need_extra_data:
        extra_data = in_stream.read(720, exception_on_overflow=False)
        extra_samples = np.frombuffer(extra_data, dtype=np.int16).astype(np.float32) / (1 << 15)
        samples = np.concatenate([samples, extra_samples])
        need_extra_data = False

    # Save input snapshot for diagnostics
    if chunk_idx == 0 or chunk_idx == NUM_CHUNKS // 2:
        input_snapshots.append(samples.copy())

    # Periodic reset
    if chunk_idx % 50 == 0 and chunk_idx != 0:
        asr_offset = 20
        vc_offset = 120

    # --- Inference (exact copy of run_rt.py logic) ---
    with torch.no_grad():
        if samples_cache is None:
            pass
        else:
            samples = np.concatenate((samples_cache, samples))
        samples_cache = samples[-samples_cache_len:]

        fbanks = extract_fbanks(samples, frame_shift=10).float()

        (encoder_output, att_cache, cnn_cache) = asr.forward_encoder_chunk(
            fbanks, asr_offset, required_cache_size, att_cache, cnn_cache)

        asr_offset += encoder_output.size(1)
        if encoder_output_cache is None:
            encoder_output = torch.cat([encoder_output[:, 0:1, :], encoder_output], dim=1)
        else:
            encoder_output = torch.cat([encoder_output_cache, encoder_output], dim=1)
        encoder_output_cache = encoder_output[:, -1:, :]
        encoder_output_upsample = encoder_output.transpose(1, 2)
        encoder_output_upsample = torch.nn.functional.interpolate(
            encoder_output_upsample, size=vc_chunk_size + 1, mode='linear', align_corners=True)
        encoder_output_upsample = encoder_output_upsample.transpose(1, 2)
        encoder_output_upsample = encoder_output_upsample[:, 1:, :]

        x = torch.randn(1, encoder_output_upsample.shape[1], 80, device='cpu', dtype=encoder_output_upsample.dtype)

        for i in range(steps):
            t = timesteps[i]
            r = timesteps[i + 1]
            t_tensor = torch.full((1,), t, device=x.device)
            r_tensor = torch.full((1,), r, device=x.device)
            u, tmp_kv_cache = vc_model(x, t_tensor, r_tensor, cache=vc_cache,
                cond=encoder_output_upsample, spks=vc_spk_emb, prompts=vc_prompt_mel,
                offset=vc_offset, kv_cache=vc_kv_cache)
            x = x - (t - r) * u

        vc_kv_cache = tmp_kv_cache
        vc_offset += x.shape[1]
        vc_cache = x

        VC_KV_CACHE_MAX_LEN = 100
        if vc_offset > 40 and vc_kv_cache[0][0].shape[2] > VC_KV_CACHE_MAX_LEN:
            for i in range(len(vc_kv_cache)):
                new_k = vc_kv_cache[i][0][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                new_v = vc_kv_cache[i][1][:, :, -VC_KV_CACHE_MAX_LEN:, :]
                vc_kv_cache[i] = (new_k, new_v)

        mel = x.transpose(1, 2)
        if vocoder_cache is not None:
            mel = torch.cat([vocoder_cache, mel], dim=-1)
        vocoder_cache = mel[:, :, -vocoder_overlap:]
        mel = (mel + 1) / 2
        wav_out = vocoder.decode(mel).squeeze()
        wav_out = wav_out.detach().cpu().numpy()

        if last_wav is not None:
            front_wav = wav_out[:vocoder_wav_overlap]
            smooth_front_wav = last_wav * down_linspace + front_wav * up_linspace
            new_wav = np.concatenate([
                smooth_front_wav,
                wav_out[vocoder_wav_overlap:-vocoder_wav_overlap]
            ], axis=0)
        else:
            new_wav = wav_out[:-vocoder_wav_overlap]
        last_wav = wav_out[-vocoder_wav_overlap:]

    # Play back
    out_stream.write(new_wav.tobytes())

    chunk_time = (time.perf_counter() - t0) * 1000
    chunk_times.append(chunk_time)
    output_wavs.append(new_wav)

    if chunk_idx % 5 == 0:
        print(f"  Chunk {chunk_idx}: time={chunk_time:.1f}ms, "
              f"in_rms={np.sqrt(np.mean(samples**2)):.3f}, "
              f"out_rms={np.sqrt(np.mean(new_wav**2)):.3f}")

# 7. Cleanup
in_stream.stop_stream()
in_stream.close()
out_stream.stop_stream()
out_stream.close()
p.terminate()

# 8. Save results
full_output = np.concatenate(output_wavs)
os.makedirs("temp", exist_ok=True)
sf.write("temp/mic_test_output.wav", full_output.astype(np.float32), 16000)

# Save first input snapshot
if input_snapshots:
    sf.write("temp/mic_test_input.wav", input_snapshots[0].astype(np.float32), 16000)

print(f"\n{'='*60}")
print(f"Results:")
print(f"  Total output: {len(full_output)} samples ({len(full_output)/16000:.2f}s)")
print(f"  Avg chunk time: {np.mean(chunk_times):.1f}ms")
print(f"  Output RMS: {np.sqrt(np.mean(full_output**2)):.4f}")
print(f"  Output peak: {np.max(np.abs(full_output)):.4f}")
print(f"  Output files saved to temp/mic_test_*.wav")

# Quick quality check
if np.sqrt(np.mean(full_output**2)) < 0.001:
    print("\n⚠️  Output is nearly silent — check mic gain or device selection!")
elif np.max(np.abs(full_output)) > 0.99:
    print("\n⚠️  Output is clipping — check mic gain!")
else:
    print("\n✅ Output levels look normal")
