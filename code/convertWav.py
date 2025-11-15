import os
import soundfile as sf
import torchaudio

# === root ===
root_dir = r"E:\2025S2\ELEC5305\TIMIT"

# === check document ===
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(".wav"):
            old_path = os.path.join(subdir, file)
            new_path = os.path.splitext(old_path)[0] + "_fixed.wav"

            try:
                
                data, samplerate = torchaudio.load(old_path)
                
                
                if samplerate != 16000:
                    resampler = torchaudio.transforms.Resample(samplerate, 16000)
                    data = resampler(data)
                    samplerate = 16000

                # mono
                if data.shape[0] > 1:
                    data = data.mean(dim=0, keepdim=True)

                # PCM_16 version
                sf.write(new_path, data.squeeze().numpy(), samplerate, subtype='PCM_16')
                print(f"Converted: {old_path} → {new_path}")

            except Exception as e:
                print(f"Failed: {old_path} | {e}")
