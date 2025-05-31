# src/feature_extraction.py
import librosa
import numpy as np
import torch
import torchaudio

# Load the pre-trained HuBERT model
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()
model.eval()  # Set to evaluation mode

def extract_features(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=bundle.sample_rate)  # HuBERT expects 16kHz
    # Pad audio if too short
    min_length = int(1.0 * sr)  # Ensure at least 1 second
    if len(y) < min_length:
        y = np.pad(y, (0, min_length - len(y)), mode='constant')
    # Convert to tensor
    waveform = torch.tensor(y).unsqueeze(0)  # Shape: (1, num_samples)

    # Extract features
    with torch.no_grad():
        features, _ = model.extract_features(waveform)  # List of feature tensors

    # Use the last layer's features (highest-level representation)
    embedding = features[-1].squeeze(0)  # Shape: (num_frames, 768)
    # Average over time to get a single embedding
    embedding = embedding.mean(dim=0).numpy()  # Shape: (768,)
    
    return embedding

def extract_features_batch(audio_paths):
    embeddings = []
    for audio_path in audio_paths:
        y, sr = librosa.load(audio_path, sr=bundle.sample_rate)
        min_length = int(1.0 * sr)
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)), mode='constant')
        waveform = torch.tensor(y).unsqueeze(0)
        embeddings.append(waveform)
    
    waveforms = torch.cat(embeddings, dim=0)  # Shape: (batch_size, num_samples)
    
    with torch.no_grad():
        features, _ = model.extract_features(waveforms)
    
    final_embeddings = []
    for i in range(len(audio_paths)):
        embedding = features[-1][i]  # Shape: (num_frames, 768)
        embedding = embedding.mean(dim=0).numpy()  # Shape: (768,)
        final_embeddings.append(embedding)
    
    return final_embeddings