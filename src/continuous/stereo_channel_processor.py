"""
Stereo Channel Processor for Continuous Learning

Processes stereo WAV files where:
- Left channel (0) = Positive class samples
- Right channel (1) = Negative class samples

Extracts audio features from both channels simultaneously for continuous training.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class StereoChannelProcessor:
    """
    Process stereo WAV files for continuous learning.

    Stereo channel assignment:
    - Left channel (index 0) → Positive class (label 1)
    - Right channel (index 1) → Negative class (label 0)
    """

    def __init__(self, config):
        """
        Initialize processor with audio configuration.

        Args:
            config: Configuration object with audio parameters
        """
        self.sample_rate = config.audio.sample_rate
        self.segment_duration = config.audio.segment_duration
        self.n_mfcc = config.audio.n_mfcc
        self.feature_types = config.audio.feature_types

        logger.info(f"StereoChannelProcessor initialized: {self.sample_rate}Hz, {self.segment_duration}s segments")

    def process_stereo_file(self,
                           wav_path: Path,
                           max_samples_per_channel: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process stereo WAV file and extract features from both channels.

        Args:
            wav_path: Path to stereo WAV file
            max_samples_per_channel: Maximum segments to extract per channel (None = all)

        Returns:
            Tuple of (X_left, y_left, X_right, y_right, offsets_left, offsets_right)
            - X_left: Features from left channel (positive class), shape (n_samples, 680)
            - y_left: Labels for left channel (all 1s), shape (n_samples,)
            - X_right: Features from right channel (negative class), shape (n_samples, 680)
            - y_right: Labels for right channel (all 0s), shape (n_samples,)
            - offsets_left: Segment start times in original file (seconds)
            - offsets_right: Segment start times in original file (seconds)
        """
        logger.info(f"Processing stereo file: {wav_path}")

        # Load stereo audio
        audio, sr = librosa.load(str(wav_path), sr=self.sample_rate, mono=False)

        # Validate stereo format
        if len(audio.shape) == 1:
            raise ValueError(f"Expected stereo audio, got mono: {wav_path}")

        if audio.shape[0] != 2:
            raise ValueError(f"Expected 2 channels, got {audio.shape[0]}: {wav_path}")

        # Separate channels
        left_channel = audio[0, :]   # Channel 0 = positive
        right_channel = audio[1, :]  # Channel 1 = negative

        duration_sec = len(left_channel) / sr
        logger.info(f"Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")

        # Extract features from each channel
        X_left, offsets_left = self._extract_features_from_channel(
            left_channel, sr, max_samples_per_channel
        )
        X_right, offsets_right = self._extract_features_from_channel(
            right_channel, sr, max_samples_per_channel
        )

        # Create labels
        y_left = np.ones(len(X_left), dtype=int)   # Positive class
        y_right = np.zeros(len(X_right), dtype=int)  # Negative class

        logger.info(f"Extracted {len(X_left)} positive samples, {len(X_right)} negative samples")
        logger.info(f"Feature dimensions: {X_left.shape[1]}D")

        return X_left, y_left, X_right, y_right, offsets_left, offsets_right

    def _extract_features_from_channel(self,
                                       channel_audio: np.ndarray,
                                       sr: int,
                                       max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from single audio channel.

        Uses sequential (non-random) segmentation to maximize data capture.
        Every second of audio is used.

        Args:
            channel_audio: Mono audio signal
            sr: Sample rate
            max_samples: Maximum number of segments (None = all possible)

        Returns:
            Tuple of (features, offsets)
            - features: (n_samples, n_features) array
            - offsets: (n_samples,) array of segment start times in seconds
        """
        segment_length = int(self.segment_duration * sr)
        total_samples = len(channel_audio)

        # Calculate number of possible segments
        n_possible_segments = total_samples // segment_length

        # Limit if requested
        if max_samples is not None:
            n_segments = min(n_possible_segments, max_samples)
        else:
            n_segments = n_possible_segments

        # For maximum data capture, use sequential segments (not random)
        # This ensures we use every second of audio
        features_list = []
        offsets_list = []

        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length

            segment = channel_audio[start_idx:end_idx]

            # Extract features
            features = self._extract_features_from_segment(segment, sr)
            features_list.append(features)

            # Store offset in seconds
            offset_sec = start_idx / sr
            offsets_list.append(offset_sec)

        return np.array(features_list), np.array(offsets_list)

    def _extract_features_from_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract features from 1-second audio segment.

        Features extracted (default config):
        - MFCC: 13 coefficients × 40 frames = 520 features
        - Spectral centroid: 40 features
        - Spectral rolloff: 40 features
        - Spectral bandwidth: 40 features
        - Zero crossing rate: 40 features
        Total: 680 features

        Args:
            segment: 1-second audio segment
            sr: Sample rate

        Returns:
            Feature vector of shape (680,) for default config
        """
        features = []

        # MFCC
        if 'mfcc' in self.feature_types:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=self.n_mfcc)
            features.append(mfcc.flatten())

        # Spectral features
        if 'spectral' in self.feature_types:
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)

            features.extend([
                spectral_centroid.flatten(),
                spectral_rolloff.flatten(),
                spectral_bandwidth.flatten(),
                zero_crossing_rate.flatten()
            ])

        # Chroma (optional)
        if 'chroma' in self.feature_types:
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            features.append(chroma.flatten())

        # Tonnetz (optional)
        if 'tonnetz' in self.feature_types:
            # Tonnetz requires harmonic/percussive source separation
            try:
                tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
                features.append(tonnetz.flatten())
            except Exception as e:
                logger.warning(f"Tonnetz extraction failed: {e}")

        # Concatenate all features
        return np.concatenate(features)


def simulate_continuous_stream(source_wav: Path,
                                chunk_duration_minutes: int,
                                config) -> List[Tuple[Path, float]]:
    """
    Simulate continuous data stream by splitting a large WAV file into chunks.

    This simulates what would happen with real-time recording:
    - Every N minutes, a new chunk is "recorded"
    - Chunks are processed as they arrive

    Useful for testing the continuous learning pipeline with existing large files.

    Args:
        source_wav: Large stereo WAV file to split
        chunk_duration_minutes: Duration of each chunk in minutes
        config: Audio configuration

    Returns:
        List of (chunk_path, timestamp) tuples
    """
    import tempfile
    import soundfile as sf
    from datetime import datetime, timedelta

    logger.info(f"Simulating continuous stream from {source_wav}")
    logger.info(f"Chunk duration: {chunk_duration_minutes} minutes")

    # Load full audio
    audio, sr = librosa.load(str(source_wav), sr=config.audio.sample_rate, mono=False)

    # Ensure stereo
    if len(audio.shape) == 1:
        raise ValueError(f"Expected stereo audio for simulation: {source_wav}")

    total_duration_sec = audio.shape[1] / sr
    chunk_duration_sec = chunk_duration_minutes * 60

    n_chunks = int(total_duration_sec / chunk_duration_sec)

    logger.info(f"Total duration: {total_duration_sec/60:.1f} minutes")
    logger.info(f"Creating {n_chunks} chunks of {chunk_duration_minutes} min each")

    # Create temporary directory for chunks
    temp_dir = Path(tempfile.mkdtemp(prefix='continuous_stream_'))
    logger.info(f"Chunk directory: {temp_dir}")

    chunks = []
    start_time = datetime.now()

    for i in range(n_chunks):
        # Extract chunk
        start_sample = int(i * chunk_duration_sec * sr)
        end_sample = int((i + 1) * chunk_duration_sec * sr)

        # Ensure we don't exceed array bounds
        end_sample = min(end_sample, audio.shape[1])

        chunk_audio = audio[:, start_sample:end_sample]

        # Save chunk
        chunk_path = temp_dir / f"chunk_{i:04d}.wav"
        sf.write(str(chunk_path), chunk_audio.T, sr)  # Transpose for soundfile

        # Simulate timestamp (chunks arriving every N minutes)
        chunk_timestamp = start_time + timedelta(minutes=i * chunk_duration_minutes)

        chunks.append((chunk_path, chunk_timestamp))

        logger.debug(f"Created chunk {i+1}/{n_chunks}: {chunk_path} ({chunk_audio.shape[1]/sr:.1f}s)")

    logger.info(f"Simulation complete: {len(chunks)} chunks created")

    return chunks
