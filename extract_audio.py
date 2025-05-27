import librosa
import numpy as np
import soundfile as sf
import os

# --- PARAMETERS ---
TARGET_SR = 44100
SEGMENT_LENGTH_SEC = 5

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MJ_MOTIF_DIR = os.path.join(BASE_DIR, "processed", "micheal_jackson", "motifs")
BEATLES_MOTIF_DIR = os.path.join(BASE_DIR, "processed", "the_beatles", "melodies")  # Repurposing as audio motifs

# Create directories
os.makedirs(MJ_MOTIF_DIR, exist_ok=True)
print(f"Created directory: {MJ_MOTIF_DIR}")
os.makedirs(BEATLES_MOTIF_DIR, exist_ok=True)
print(f"Created directory: {BEATLES_MOTIF_DIR}")

# --- SEGMENT & SAVE MOTIFS FUNCTION ---
def segment_and_save(y, sr, segment_length_sec, out_dir, prefix):
    segment_samples = segment_length_sec * sr
    print(f"Audio length: {len(y)} samples, Segment length: {segment_samples} samples")
    for i in range(0, len(y), segment_samples):
        segment = y[i:i + segment_samples]
        print(f"Processing segment {i//segment_samples}, length: {len(segment)} samples")
        if len(segment) == segment_samples:
            out_path = os.path.join(out_dir, f"{prefix}_motif_{i//segment_samples}.wav")
            sf.write(out_path, segment, sr)
            print(f"Saved: {out_path}")
        else:
            print(f"Skipped segment {i//segment_samples}: too short ({len(segment)} samples)")

# --- LOAD AUDIO (Michael Jackson) ---
# Adjust the path based on your directory; assuming the file is directly in MICHEAL_JACKSON/
mj_audio_path = os.path.join(BASE_DIR, "data", "Micheal_Jackson", "stems", "Micheal_Jackson_The_Lady_In_My_Life.wav")
print(f"Loading Michael Jackson audio: {mj_audio_path}")
try:
    y_mj, sr_mj = librosa.load(mj_audio_path, sr=None)
except Exception as e:
    print(f"Error loading Michael Jackson audio: {e}")
    exit(1)

# Standardize sample rate
if sr_mj != TARGET_SR:
    print(f"Resampling Michael Jackson audio from {sr_mj} Hz to {TARGET_SR} Hz")
    y_mj = librosa.resample(y_mj, orig_sr=sr_mj, target_sr=TARGET_SR)
    sr_mj = TARGET_SR
# Normalize
y_mj = y_mj / np.max(np.abs(y_mj))
print(f"Michael Jackson audio loaded: {len(y_mj)} samples at {sr_mj} Hz")

# Segment and save Michael Jackson motifs
segment_and_save(y_mj, sr_mj, SEGMENT_LENGTH_SEC, MJ_MOTIF_DIR, "mj")

# --- LOAD AUDIO (The Beatles) ---
# Adjust the path to the MP3 file
beatles_audio_path = os.path.join(BASE_DIR, "data", "The_Beatles", "stems", "The_Beatles_Please_Mr._Postman.mp3")
print(f"Loading The Beatles audio: {beatles_audio_path}")
try:
    y_beatles, sr_beatles = librosa.load(beatles_audio_path, sr=None)
except Exception as e:
    print(f"Error loading The Beatles audio: {e}")
    exit(1)

# Standardize sample rate
if sr_beatles != TARGET_SR:
    print(f"Resampling The Beatles audio from {sr_beatles} Hz to {TARGET_SR} Hz")
    y_beatles = librosa.resample(y_beatles, orig_sr=sr_beatles, target_sr=TARGET_SR)
    sr_beatles = TARGET_SR
# Normalize
y_beatles = y_beatles / np.max(np.abs(y_beatles))
print(f"The Beatles audio loaded: {len(y_beatles)} samples at {sr_beatles} Hz")

# Segment and save The Beatles motifs
segment_and_save(y_beatles, sr_beatles, SEGMENT_LENGTH_SEC, BEATLES_MOTIF_DIR, "beatles")