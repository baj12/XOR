import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import Config, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio file processing and feature extraction"""
    
    def __init__(self, audio_config):
        self.config = audio_config  # Use the AudioConfig object directly
        self.segment_samples = int(self.config.sample_rate * self.config.segment_duration)
        self.scaler = StandardScaler()
        
    def load_audio_info(self, filepath: str) -> Tuple[int, float]:
        """Get audio file info without loading entire file"""
        info = sf.info(filepath)
        return info.frames, info.duration
    
    def load_audio_segment(self, filepath: str, start_frame: int, num_frames: int) -> np.ndarray:
        """Load a specific segment from audio file"""
        try:
            audio, _ = sf.read(filepath, start=start_frame, frames=num_frames)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            # Resample if necessary
            file_sr = sf.info(filepath).samplerate
            if file_sr != self.config.sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=file_sr,
                    target_sr=self.config.sample_rate
                )
            
            return audio
        except Exception as e:
            logger.error(f"Error loading audio segment from {filepath}: {e}")
            return np.zeros(self.segment_samples)
    
    def extract_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """Extract features from audio segment"""
        features = []
        
        # Ensure segment is correct length
        if len(audio_segment) < self.segment_samples:
            # Pad with zeros
            audio_segment = np.pad(audio_segment, (0, self.segment_samples - len(audio_segment)))
        elif len(audio_segment) > self.segment_samples:
            # Truncate
            audio_segment = audio_segment[:self.segment_samples]
        
        if 'mfcc' in self.config.feature_types:
            mfcc = librosa.feature.mfcc(
                y=audio_segment,
                sr=self.config.sample_rate,
                n_mfcc=self.config.n_mfcc,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            features.append(mfcc.flatten())
        
        if 'spectral' in self.config.feature_types:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_segment, sr=self.config.sample_rate, hop_length=self.config.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_segment, sr=self.config.sample_rate, hop_length=self.config.hop_length
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_segment, sr=self.config.sample_rate, hop_length=self.config.hop_length
            )
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                audio_segment, hop_length=self.config.hop_length
            )
            
            features.extend([
                spectral_centroids.flatten(),
                spectral_rolloff.flatten(),
                spectral_bandwidth.flatten(),
                zero_crossing_rate.flatten()
            ])
        
        if 'chroma' in self.config.feature_types:
            chroma = librosa.feature.chroma_stft(
                y=audio_segment,
                sr=self.config.sample_rate,
                hop_length=self.config.hop_length
            )
            features.append(chroma.flatten())
        
        if 'tonnetz' in self.config.feature_types:
            tonnetz = librosa.feature.tonnetz(
                y=audio_segment,
                sr=self.config.sample_rate
            )
            features.append(tonnetz.flatten())
        
        # Concatenate all features
        if features:
            feature_vector = np.concatenate(features)
        else:
            # If no features specified, use basic spectral features
            mfcc = librosa.feature.mfcc(
                y=audio_segment,
                sr=self.config.sample_rate,
                n_mfcc=13,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            feature_vector = mfcc.flatten()
            
        return feature_vector
    
    def create_preprocessed_data(self, audio_files: List[str], labels: List[int], 
                                output_path: str, samples_per_file: int = 1000):
        """Create preprocessed data file for random access"""
        logger.info(f"Creating preprocessed data with {samples_per_file} samples per file")
        
        all_features = []
        all_labels = []
        
        for file_path, label in zip(audio_files, labels):
            logger.info(f"Processing {file_path} (label: {label})")
            
            # Get file info
            total_frames, duration = self.load_audio_info(file_path)
            
            # Calculate possible start positions for 1-second segments
            max_start_frame = max(0, total_frames - self.segment_samples)
            
            if max_start_frame <= 0:
                logger.warning(f"File {file_path} is too short for 1-second segments")
                continue
            
            # Generate random start positions
            start_frames = np.random.randint(0, max_start_frame, samples_per_file)
            
            for start_frame in tqdm(start_frames, desc=f"Processing {os.path.basename(file_path)}"):
                # Load audio segment
                audio_segment = self.load_audio_segment(file_path, start_frame, self.segment_samples)
                
                # Extract features
                features = self.extract_features(audio_segment)
                
                all_features.append(features)
                all_labels.append(label)
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Normalize features
        X_normalized = self.scaler.fit_transform(X)
        
        # Save preprocessed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, 
                          features=X_normalized, 
                          labels=y,
                          scaler_mean=self.scaler.mean_,
                          scaler_scale=self.scaler.scale_,
                          feature_dim=X_normalized.shape[1])
        
        logger.info(f"Preprocessed data saved to {output_path}")
        logger.info(f"Feature dimensions: {X_normalized.shape}")
        
        return X_normalized, y


class AudioDataGenerator:
    """Main class for generating audio training data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = AudioProcessor(config.audio)  # Pass the AudioConfig object directly
    
    def generate_from_files(self, audio_files: List[str], labels: List[int]) -> pd.DataFrame:
        """Generate training data from audio files"""
        if len(audio_files) != len(labels):
            raise ValueError("Number of audio files must match number of labels")
        
        # Create preprocessed data
        X, y = self.processor.create_preprocessed_data(
            audio_files, 
            labels, 
            f"data/preprocessed/{self.config.experiment.id}_preprocessed.npz",
            samples_per_file=self.config.data.samples_per_file
        )
        
        # Create DataFrame
        feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_columns)
        df['label'] = y
        
        return df

def generate_audio_data_from_config(config_path: str, audio_files: List[str], labels: List[int]) -> str:
    """Generate audio data based on configuration"""
    config = load_config(config_path)
    
    generator = AudioDataGenerator(config)
    df = generator.generate_from_files(audio_files, labels)
    
    # Save data
    os.makedirs('data/raw', exist_ok=True)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    output_path = f"data/raw/{config_name}_data.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Audio data saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate audio dataset")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--audio-files', type=str, nargs='+', required=True, 
                       help='Paths to audio files')
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                       help='Labels for audio files (0 or 1)')
    
    args = parser.parse_args()
    
    if len(args.audio_files) != len(args.labels):
        raise ValueError("Number of audio files must match number of labels")
    
    generate_audio_data_from_config(args.config, args.audio_files, args.labels)
    