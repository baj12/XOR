import hashlib
import json  # Add this line at the top with other imports
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils import Config, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFileAnalyzer:
    """Analyze and validate audio files for compatibility"""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_file(self, filepath: str) -> Dict:
        """Comprehensive analysis of audio file"""
        logger.info(f"Analyzing audio file: {filepath}")
        
        # Expand user path
        filepath = os.path.expanduser(filepath)
        
        # Basic file info
        file_info = sf.info(filepath)
        file_stats = os.stat(filepath)
        
        # Load audio for analysis
        audio, sr = librosa.load(filepath, sr=None)
        
        # Calculate MD5 hash
        with open(filepath, 'rb') as f:
            # md5_hash = hashlib.md5(f.read()).hexdigest()
            md5_hash = self._calculate_file_hash(filepath)

            # Initialize analysis dict with file metadata
        analysis = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'file_size_mb': file_stats.st_size / (1024 * 1024),
            'md5_hash': md5_hash,
            
            # Audio properties
            'sample_rate': file_info.samplerate,
            'duration_seconds': file_info.duration,
            'channels': file_info.channels,
            'frames': file_info.frames,
            'format': file_info.format,
            'subtype': file_info.subtype,
        }
        
        # Load audio in chunks for analysis instead of all at once
        chunk_size = 60 * file_info.samplerate  # Process 60 seconds at a time
        
        # Initialize accumulators for statistics
        audio_min, audio_max = float('inf'), float('-inf')
        audio_sum, audio_sum_squared = 0.0, 0.0
        zero_crossings = 0
        silent_samples = 0
        sample_count = 0
        
        # Process audio in chunks
        for i in range(0, file_info.frames, chunk_size):
            # Load chunk
            frames_to_read = min(chunk_size, file_info.frames - i)
            chunk, sr = sf.read(filepath, start=i, frames=frames_to_read)
            
            if chunk.ndim > 1:
                chunk = np.mean(chunk, axis=1)  # Convert to mono if needed
            
            # Update statistics
            audio_min = min(audio_min, np.min(chunk))
            audio_max = max(audio_max, np.max(chunk))
            audio_sum += np.sum(chunk)
            audio_sum_squared += np.sum(chunk**2)
            
            # Count zero crossings and silence
            zero_crossings += np.sum(librosa.zero_crossings(chunk))
            silent_samples += np.sum(np.abs(chunk) < 0.01)
            sample_count += len(chunk)
            
            # Force garbage collection after processing chunk
            del chunk
            import gc
            gc.collect()
        
        # Calculate final statistics
        audio_mean = audio_sum / sample_count
        audio_std = np.sqrt((audio_sum_squared / sample_count) - (audio_mean**2))
        audio_rms = np.sqrt(audio_sum_squared / sample_count)
        
        # Update analysis with calculated statistics
        analysis.update({
            'audio_min': float(audio_min),
            'audio_max': float(audio_max),
            'audio_mean': float(audio_mean),
            'audio_std': float(audio_std),
            'audio_rms': float(audio_rms),
            'zero_crossing_rate': float(zero_crossings / sample_count),
            'silence_percentage': float(silent_samples / sample_count * 100),
        })
        
        # Calculate spectral features using a sample of the audio
        self._add_spectral_features(analysis, filepath)
        
        self.analysis_results[filepath] = analysis
        return analysis

    
    def _calculate_file_hash(self, filepath, block_size=65536):
        """Calculate MD5 hash in chunks to avoid loading entire file"""
        h = hashlib.md5()
        with open(filepath, 'rb') as f:
            for block in iter(lambda: f.read(block_size), b''):
                h.update(block)
        return h.hexdigest()
    
    def _add_spectral_features(self, analysis, filepath):
        """Add spectral features using a sample of the audio"""
        # Load only a 30-second sample for spectral analysis
        duration = min(30.0, analysis['duration_seconds'])
        audio, sr = librosa.load(filepath, sr=analysis['sample_rate'], duration=duration)
        
        # Calculate spectral features
        analysis['spectral_centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        analysis['spectral_bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        analysis['spectral_rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        
        # Calculate peak features
        analysis['peak_count'] = len(librosa.onset.onset_detect(y=audio, sr=sr))
        
        # Calculate dynamic range
        analysis['dynamic_range_db'] = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
        
        del audio
        import gc
        gc.collect()
    
    def _calculate_silence_percentage(self, audio, threshold=0.01):
        """Calculate percentage of audio that is effectively silent"""
        silent_samples = np.sum(np.abs(audio) < threshold)
        return (silent_samples / len(audio)) * 100
    
    def compare_files(self, file1_analysis: Dict, file2_analysis: Dict) -> Dict:
        """Compare two audio files for compatibility"""
        compatibility = {
            'sample_rate_match': file1_analysis['sample_rate'] == file2_analysis['sample_rate'],
            'channels_match': file1_analysis['channels'] == file2_analysis['channels'],
            'format_match': file1_analysis['format'] == file2_analysis['format'],
            
            # Duration comparison
            'duration_ratio': file1_analysis['duration_seconds'] / file2_analysis['duration_seconds'],
            'duration_difference_sec': abs(file1_analysis['duration_seconds'] - file2_analysis['duration_seconds']),
            
            # Audio statistics comparison
            'rms_ratio': file1_analysis['audio_rms'] / (file2_analysis['audio_rms'] + 1e-10),
            'dynamic_range_diff': abs(file1_analysis['dynamic_range_db'] - file2_analysis['dynamic_range_db']),
            'spectral_centroid_ratio': file1_analysis['spectral_centroid_mean'] / (file2_analysis['spectral_centroid_mean'] + 1e-10),
            
            # Quality indicators
            'silence_diff': abs(file1_analysis['silence_percentage'] - file2_analysis['silence_percentage']),
            'zero_crossing_diff': abs(file1_analysis['zero_crossing_rate'] - file2_analysis['zero_crossing_rate']),
        }
        
        # Overall compatibility score
        score = 0
        if compatibility['sample_rate_match']: score += 25
        if compatibility['channels_match']: score += 25
        if compatibility['format_match']: score += 10
        if 0.5 <= compatibility['duration_ratio'] <= 2.0: score += 15
        if 0.1 <= compatibility['rms_ratio'] <= 10.0: score += 10
        if compatibility['dynamic_range_diff'] < 20: score += 10
        if compatibility['silence_diff'] < 20: score += 5
        
        compatibility['compatibility_score'] = score
        compatibility['is_compatible'] = score >= 75
        
        return compatibility
    
    def create_qc_report(self, analyses: List[Dict], compatibility: Dict, output_dir: str):
        """Create comprehensive QC report with plots and tables"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Summary table
        self._create_summary_table(analyses, compatibility, output_dir)
        
        # 2. Audio comparison plots
        self._create_comparison_plots(analyses, output_dir)
        
        # 3. Spectrograms
        self._create_spectrograms(analyses, output_dir)
        
        # 4. Feature comparison
        self._create_feature_comparison(analyses, output_dir)
        
        # 5. Compatibility report
        self._create_compatibility_report(compatibility, output_dir)
    
    def _create_summary_table(self, analyses: List[Dict], compatibility: Dict, output_dir: str):
        """Create summary table"""
        summary_data = []
        for i, analysis in enumerate(analyses):
            summary_data.append({
                'Class': f'Class {i}',
                'Filename': analysis['filename'],
                'Duration (s)': f"{analysis['duration_seconds']:.2f}",
                'Sample Rate': analysis['sample_rate'],
                'Channels': analysis['channels'],
                'File Size (MB)': f"{analysis['file_size_mb']:.2f}",
                'RMS Level': f"{analysis['audio_rms']:.4f}",
                'Dynamic Range (dB)': f"{analysis['dynamic_range_db']:.1f}",
                'Silence %': f"{analysis['silence_percentage']:.1f}",
                'Spectral Centroid': f"{analysis['spectral_centroid_mean']:.0f} Hz"
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{output_dir}/audio_summary.csv", index=False)
        
        # Create formatted table plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center', fontsize=10)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code compatibility
        if compatibility.get('is_compatible', False):
            table[(1, 0)].set_facecolor('#90EE90')  # Light green
            table[(2, 0)].set_facecolor('#90EE90')
        else:
            table[(1, 0)].set_facecolor('#FFB6C1')  # Light red
            table[(2, 0)].set_facecolor('#FFB6C1')
        
        plt.title('Audio Files Summary', fontsize=14, fontweight='bold')
        plt.savefig(f"{output_dir}/audio_summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary table saved to {output_dir}/audio_summary.csv")

    def _create_comparison_plots(self, analyses: List[Dict], output_dir: str):
        """Create audio waveform comparison plots with memory optimization"""
        fig, axes = plt.subplots(len(analyses), 2, figsize=(15, 4*len(analyses)))
        if len(analyses) == 1:
            axes = axes.reshape(1, -1)
        
        for i, analysis in enumerate(analyses):
            try:
                logger.info(f"Creating comparison plots for {analysis['filename']}...")
                
                # Load only first 10 seconds for waveform visualization
                max_duration = min(10.0, analysis['duration_seconds'])
                audio, sr = librosa.load(analysis['filepath'], sr=None, duration=max_duration)
                
                # Downsample for visualization if too many samples
                if len(audio) > 220500:  # More than 10 seconds at 22kHz
                    downsample_factor = len(audio) // 220500
                    audio = audio[::downsample_factor]
                    sr = sr // downsample_factor
                
                # Time axis
                time = np.linspace(0, len(audio)/sr, len(audio))
                
                # Waveform
                axes[i, 0].plot(time, audio, alpha=0.7, linewidth=0.5)
                axes[i, 0].set_title(f"Class {i}: {analysis['filename']} - Waveform (first {max_duration:.0f}s)")
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel('Amplitude')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Histogram of amplitudes (use sample for large files)
                audio_sample = audio[::max(1, len(audio)//10000)]  # Sample for histogram
                axes[i, 1].hist(audio_sample, bins=50, alpha=0.7, density=True)
                axes[i, 1].axvline(analysis['audio_mean'], color='red', linestyle='--', 
                                label=f'Mean: {analysis["audio_mean"]:.3f}')
                axes[i, 1].axvline(analysis['audio_mean'] + analysis['audio_std'], 
                                color='orange', linestyle='--', alpha=0.7,
                                label=f'±1σ: {analysis["audio_std"]:.3f}')
                axes[i, 1].axvline(analysis['audio_mean'] - analysis['audio_std'], 
                                color='orange', linestyle='--', alpha=0.7)
                axes[i, 1].set_title(f"Class {i}: Amplitude Distribution")
                axes[i, 1].set_xlabel('Amplitude')
                axes[i, 1].set_ylabel('Density')
                axes[i, 1].legend()
                axes[i, 1].grid(True, alpha=0.3)
                
                # Force garbage collection
                del audio
                
            except Exception as e:
                logger.error(f"Error creating comparison plots for {analysis['filename']}: {e}")
                axes[i, 0].text(0.5, 0.5, f'Error loading\n{analysis["filename"]}', 
                            transform=axes[i, 0].transAxes, ha='center', va='center')
                axes[i, 1].text(0.5, 0.5, f'Error loading\n{analysis["filename"]}', 
                            transform=axes[i, 1].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/audio_comparison.png", dpi=150, bbox_inches='tight')  # Reduced DPI
        plt.close()
        
        logger.info(f"Audio comparison plots saved to {output_dir}/audio_comparison.png")

    
    def _create_spectrograms(self, analyses: List[Dict], output_dir: str):
        """Create spectrogram comparisons with memory optimization"""
        fig, axes = plt.subplots(len(analyses), 1, figsize=(12, 4*len(analyses)))
        if len(analyses) == 1:
            axes = [axes]
        
        for i, analysis in enumerate(analyses):
            try:
                logger.info(f"Creating spectrogram for {analysis['filename']}...")
                
                # Load audio with memory optimization
                # Load only first 30 seconds to avoid memory issues
                max_duration = min(30.0, analysis['duration_seconds'])
                max_samples = int(max_duration * analysis['sample_rate'])
                
                audio, sr = librosa.load(analysis['filepath'], sr=None, duration=max_duration)
                logger.info(f"Loaded {len(audio)} samples ({len(audio)/sr:.1f}s) for spectrogram")
                
                # Create spectrogram with reduced resolution for memory efficiency
                n_fft = min(2048, len(audio) // 4)  # Adjust n_fft based on audio length
                hop_length = n_fft // 4
                
                D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
                
                img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[i])
                axes[i].set_title(f"Class {i}: {analysis['filename']} - Spectrogram (first {max_duration:.0f}s)")
                plt.colorbar(img, ax=axes[i], format="%+2.f dB")
                
                # Force garbage collection
                del audio, D
                
            except Exception as e:
                logger.error(f"Error creating spectrogram for {analysis['filename']}: {e}")
                # Create empty plot on error
                axes[i].text(0.5, 0.5, f'Error loading\n{analysis["filename"]}', 
                            transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(f"Class {i}: {analysis['filename']} - Error")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/spectrograms.png", dpi=150, bbox_inches='tight')  # Reduced DPI
        plt.close()
        
        logger.info(f"Spectrograms saved to {output_dir}/spectrograms.png")

    def _create_feature_comparison(self, analyses: List[Dict], output_dir: str):
        """Create feature comparison radar chart"""
        features = ['audio_rms', 'zero_crossing_rate', 'spectral_centroid_mean', 
                   'spectral_bandwidth_mean', 'dynamic_range_db', 'silence_percentage']
        
        feature_labels = ['RMS Level', 'Zero Crossing', 'Spectral Centroid', 
                         'Spectral Bandwidth', 'Dynamic Range', 'Silence %']
        
        # Extract and normalize features
        feature_data = []
        for analysis in analyses:
            values = [analysis[f] for f in features]
            feature_data.append(values)
        
        # Normalize to 0-1 range
        feature_array = np.array(feature_data)
        feature_min = np.min(feature_array, axis=0)
        feature_max = np.max(feature_array, axis=0)
        feature_range = feature_max - feature_min
        feature_range[feature_range == 0] = 1
        
        normalized_features = (feature_array - feature_min) / feature_range
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (normalized_vals, analysis) in enumerate(zip(normalized_features, analyses)):
            values = normalized_vals.tolist()
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f"Class {i}: {analysis['filename']}", 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.set_title('Audio Feature Comparison (Normalized)', size=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature radar chart saved to {output_dir}/feature_radar.png")
    
    def _create_compatibility_report(self, compatibility: Dict, output_dir: str):
        """Create compatibility assessment report"""
        if 'compatibility_score' not in compatibility:
            # Skip if no compatibility analysis (e.g., more than 2 files)
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Compatibility score
        score = compatibility['compatibility_score']
        colors = ['red' if score < 50 else 'orange' if score < 75 else 'green']
        ax1.bar(['Compatibility Score'], [score], color=colors[0], alpha=0.7)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Score')
        ax1.set_title(f'Overall Compatibility: {score}/100')
        ax1.grid(True, alpha=0.3)
        
        # Binary checks
        checks = ['Sample Rate Match', 'Channels Match', 'Format Match']
        values = [compatibility['sample_rate_match'], compatibility['channels_match'], 
                 compatibility['format_match']]
        colors_check = ['green' if v else 'red' for v in values]
        
        ax2.bar(checks, [1 if v else 0 for v in values], color=colors_check, alpha=0.7)
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Match')
        ax2.set_title('Basic Compatibility Checks')
        ax2.tick_params(axis='x', rotation=45)
        
        # Ratio comparisons
        ratios = ['Duration Ratio', 'RMS Ratio', 'Spectral Centroid Ratio']
        ratio_values = [compatibility['duration_ratio'], compatibility['rms_ratio'], 
                       compatibility['spectral_centroid_ratio']]
        
        ax3.bar(ratios, ratio_values, alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Ideal Ratio')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Audio Property Ratios')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Difference measures
        diffs = ['Dynamic Range Diff', 'Silence Diff', 'Zero Crossing Diff']
        diff_values = [compatibility['dynamic_range_diff'], compatibility['silence_diff'], 
                      compatibility['zero_crossing_diff']]
        
        ax4.bar(diffs, diff_values, alpha=0.7)
        ax4.set_ylabel('Difference')
        ax4.set_title('Audio Property Differences')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/compatibility_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save compatibility data as JSON
        with open(f"{output_dir}/compatibility_data.json", 'w') as f:
            json.dump(compatibility, f, indent=2)
        
        logger.info(f"Compatibility report saved to {output_dir}/compatibility_report.png")


class AudioProcessor:
    """Handles audio file processing and feature extraction"""
    
    def __init__(self, audio_config):
        self.config = audio_config
        self.segment_samples = int(self.config.sample_rate * self.config.segment_duration)
        self.scaler = StandardScaler()
        
    def load_audio_info(self, filepath: str) -> Tuple[int, float]:
        """Get audio file info without loading entire file"""
        filepath = os.path.expanduser(filepath)
        info = sf.info(filepath)
        return info.frames, info.duration
    
    def load_audio_segment(self, filepath: str, start_frame: int, num_frames: int) -> np.ndarray:
        """Load a specific segment from audio file"""
        try:
            filepath = os.path.expanduser(filepath)
            audio, _ = sf.read(filepath, start=start_frame, frames=num_frames)
            
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
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
        
        if len(audio_segment) < self.segment_samples:
            audio_segment = np.pad(audio_segment, (0, self.segment_samples - len(audio_segment)))
        elif len(audio_segment) > self.segment_samples:
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
        
        if features:
            feature_vector = np.concatenate(features)
        else:
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
            
            total_frames, duration = self.load_audio_info(file_path)
            max_start_frame = max(0, total_frames - self.segment_samples)
            
            if max_start_frame <= 0:
                logger.warning(f"File {file_path} is too short for 1-second segments")
                continue
            
            start_frames = np.random.randint(0, max_start_frame, samples_per_file)
            
            for start_frame in tqdm(start_frames, desc=f"Processing {os.path.basename(file_path)}"):
                audio_segment = self.load_audio_segment(file_path, start_frame, self.segment_samples)
                features = self.extract_features(audio_segment)
                
                all_features.append(features)
                all_labels.append(label)
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        X_normalized = self.scaler.fit_transform(X)
        
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
    """Main class for generating audio training data with comprehensive QC"""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = AudioFileAnalyzer()
        self.processor = AudioProcessor(config.audio)
    
    def generate_from_files(self, class_files_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """Generate training data from multiple audio files per class with comprehensive validation and QC"""
        
        # Flatten the structure for processing
        all_audio_files = []
        all_labels = []
        
        for class_name, file_paths in class_files_dict.items():
            class_label = int(class_name.split('_')[1])  # Extract number from 'class_0', 'class_1', etc.
            
            for file_path in file_paths:
                all_audio_files.append(file_path)
                all_labels.append(class_label)
        
        logger.info(f"Processing {len(all_audio_files)} files across {len(class_files_dict)} classes")
        for class_name, file_paths in class_files_dict.items():
            logger.info(f"  {class_name}: {len(file_paths)} files")
        
        # Set up QC output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        qc_output_dir = f"qc_reports/{self.config.experiment.id}_{timestamp}"
        os.makedirs(qc_output_dir, exist_ok=True)
        
        # Step 1: Analyze all audio files
        logger.info("Step 1: Analyzing audio files...")
        analyses = []
        analyses_by_class = {}
        
        for class_name, file_paths in class_files_dict.items():
            analyses_by_class[class_name] = []
            for filepath in file_paths:
                analysis = self.analyzer.analyze_file(filepath)
                analyses.append(analysis)
                analyses_by_class[class_name].append(analysis)
        
        # Step 2: Check compatibility within and across classes
        logger.info("Step 2: Checking compatibility...")
        compatibility_report = self._check_multi_file_compatibility(analyses_by_class)
        
        # Step 3: Create comprehensive QC report
        logger.info("Step 3: Creating QC report...")
        self._create_multi_file_qc_report(analyses_by_class, compatibility_report, qc_output_dir)
        
        # Step 4: Update configuration with audio file info
        logger.info("Step 4: Updating configuration...")
        updated_config = self._update_config_with_multi_audio_info(analyses_by_class)
        
        # Save updated config
        config_path = f"{qc_output_dir}/updated_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(updated_config, f, default_flow_style=False)
        
        # Step 5: Generate training data
        logger.info("Step 5: Generating training data...")
        X, y = self.processor.create_preprocessed_data(
            all_audio_files, 
            all_labels, 
            f"{qc_output_dir}/preprocessed_data.npz",
            samples_per_file=self.config.data.samples_per_file
        )
        
        # Create DataFrame
        feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_columns)
        df['label'] = y
        
        # Step 6: Final data validation
        logger.info("Step 6: Final data validation...")
        self._validate_generated_data(df, qc_output_dir)
        
        logger.info(f"QC report and data generated successfully in: {qc_output_dir}")
        return df

    def _check_multi_file_compatibility(self, analyses_by_class: Dict[str, List[Dict]]) -> Dict:
        """Check compatibility within and across classes"""
        compatibility_report = {
            'within_class': {},
            'across_class': {},
            'overall_compatible': True,
            'issues': []
        }
        
        # Check within each class
        for class_name, class_analyses in analyses_by_class.items():
            if len(class_analyses) > 1:
                within_class_issues = []
                
                # Check all pairs within class
                for i in range(len(class_analyses)):
                    for j in range(i + 1, len(class_analyses)):
                        comp = self.analyzer.compare_files(class_analyses[i], class_analyses[j])
                        if not comp['is_compatible']:
                            within_class_issues.append(f"Files {i} and {j} not compatible (score: {comp['compatibility_score']})")
                
                compatibility_report['within_class'][class_name] = {
                    'compatible': len(within_class_issues) == 0,
                    'issues': within_class_issues
                }
                
                if within_class_issues:
                    compatibility_report['overall_compatible'] = False
                    compatibility_report['issues'].extend([f"{class_name}: {issue}" for issue in within_class_issues])
        
        # Check across classes (sample comparison)
        class_names = list(analyses_by_class.keys())
        if len(class_names) >= 2:
            # Compare first file from each class
            sample_analyses = [analyses_by_class[class_name][0] for class_name in class_names[:2]]
            across_comp = self.analyzer.compare_files(sample_analyses[0], sample_analyses[1])
            
            compatibility_report['across_class'] = {
                'sample_compatibility': across_comp,
                'recommendation': 'Good' if across_comp['is_compatible'] else 'Review needed'
            }
        
        return compatibility_report

    def _create_multi_file_qc_report(self, analyses_by_class: Dict[str, List[Dict]], 
                                        compatibility_report: Dict, output_dir: str):
        """Create QC report for multiple files per class"""
        
        # Create summary table for all files
        summary_data = []
        for class_name, class_analyses in analyses_by_class.items():
            for i, analysis in enumerate(class_analyses):
                summary_data.append({
                    'Class': class_name,
                    'File_Index': i,
                    'Filename': analysis['filename'],
                    'Duration (s)': f"{analysis['duration_seconds']:.2f}",
                    'Sample Rate': analysis['sample_rate'],
                    'Channels': analysis['channels'],
                    'File Size (MB)': f"{analysis['file_size_mb']:.2f}",
                    'RMS Level': f"{analysis['audio_rms']:.4f}",
                    'Dynamic Range (dB)': f"{analysis['dynamic_range_db']:.1f}",
                    'Silence %': f"{analysis['silence_percentage']:.1f}",
                    'Spectral Centroid': f"{analysis['spectral_centroid_mean']:.0f} Hz"
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{output_dir}/multi_file_summary.csv", index=False)
        
        # Create class-wise statistics
        class_stats = {}
        for class_name, class_analyses in analyses_by_class.items():
            stats = {
                'file_count': len(class_analyses),
                'total_duration': sum(a['duration_seconds'] for a in class_analyses),
                'avg_duration': np.mean([a['duration_seconds'] for a in class_analyses]),
                'avg_rms': np.mean([a['audio_rms'] for a in class_analyses]),
                'avg_dynamic_range': np.mean([a['dynamic_range_db'] for a in class_analyses]),
                'sample_rates': list(set(a['sample_rate'] for a in class_analyses)),
                'channels': list(set(a['channels'] for a in class_analyses)),
            }
            class_stats[class_name] = stats
        
        # Save class statistics
        with open(f"{output_dir}/class_statistics.json", 'w') as f:
            json.dump(class_stats, f, indent=2)
        
        # Create visualization plots
        self._create_multi_file_plots(analyses_by_class, class_stats, output_dir)
        
        # Save compatibility report
        with open(f"{output_dir}/compatibility_report.json", 'w') as f:
            json.dump(compatibility_report, f, indent=2)
        
        logger.info(f"Multi-file QC report created in {output_dir}")

    def _create_multi_file_plots(self, analyses_by_class: Dict[str, List[Dict]], 
                            class_stats: Dict, output_dir: str):
        """Create plots for multiple files per class"""
        
        # 1. Class overview plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # File count per class
        classes = list(class_stats.keys())
        file_counts = [class_stats[c]['file_count'] for c in classes]
        axes[0, 0].bar(classes, file_counts, alpha=0.7)
        axes[0, 0].set_title('Files per Class')
        axes[0, 0].set_ylabel('File Count')
        
        # Total duration per class
        total_durations = [class_stats[c]['total_duration'] for c in classes]
        axes[0, 1].bar(classes, total_durations, alpha=0.7, color='orange')
        axes[0, 1].set_title('Total Duration per Class')
        axes[0, 1].set_ylabel('Duration (seconds)')
        
        # Average RMS per class
        avg_rms = [class_stats[c]['avg_rms'] for c in classes]
        axes[1, 0].bar(classes, avg_rms, alpha=0.7, color='green')
        axes[1, 0].set_title('Average RMS Level per Class')
        axes[1, 0].set_ylabel('RMS Level')
        
        # Dynamic range comparison
        avg_dynamic_range = [class_stats[c]['avg_dynamic_range'] for c in classes]
        axes[1, 1].bar(classes, avg_dynamic_range, alpha=0.7, color='red')
        axes[1, 1].set_title('Average Dynamic Range per Class')
        axes[1, 1].set_ylabel('Dynamic Range (dB)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Individual file plots (sample from each class)
        max_files_to_plot = 3
        for class_name, class_analyses in analyses_by_class.items():
            files_to_plot = class_analyses[:max_files_to_plot]
            
            if len(files_to_plot) > 1:
                fig, axes = plt.subplots(len(files_to_plot), 2, figsize=(15, 4*len(files_to_plot)))
                if len(files_to_plot) == 1:
                    axes = axes.reshape(1, -1)
                    
                for i, analysis in enumerate(files_to_plot):
                    try:
                        # Load and plot waveform
                        audio, sr = librosa.load(analysis['filepath'], sr=None, duration=10.0)
                        time = np.linspace(0, len(audio)/sr, len(audio))
                        
                        axes[i, 0].plot(time, audio, alpha=0.7, linewidth=0.5)
                        axes[i, 0].set_title(f"{class_name} - File {i}: {analysis['filename']}")
                        axes[i, 0].set_xlabel('Time (s)')
                        axes[i, 0].set_ylabel('Amplitude')
                        axes[i, 0].grid(True, alpha=0.3)
                        
                        # Amplitude histogram
                        axes[i, 1].hist(audio[::max(1, len(audio)//10000)], bins=50, alpha=0.7, density=True)
                        axes[i, 1].set_title(f"Amplitude Distribution")
                        axes[i, 1].set_xlabel('Amplitude')
                        axes[i, 1].set_ylabel('Density')
                        axes[i, 1].grid(True, alpha=0.3)
                        
                    except Exception as e:
                        logger.error(f"Error plotting {analysis['filename']}: {e}")
                        
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{class_name}_files_comparison.png", dpi=150, bbox_inches='tight')
                plt.close()

    def _update_config_with_multi_audio_info(self, analyses_by_class: Dict[str, List[Dict]]) -> Dict:
        """Update configuration with multiple audio files information"""
        config_dict = {
            'experiment': vars(self.config.experiment),
            'data': vars(self.config.data),
            'audio': vars(self.config.audio),
            'ga': vars(self.config.ga),
            'model': vars(self.config.model),
            'metrics': self.config.metrics
        }
        
        # Add audio file info for multiple files per class
        config_dict['audio']['audio_files'] = {}
        
        for class_name, class_analyses in analyses_by_class.items():
            config_dict['audio']['audio_files'][class_name] = {
                'paths': [analysis['filepath'] for analysis in class_analyses],
                'file_count': len(class_analyses),
                'total_duration': sum(a['duration_seconds'] for a in class_analyses),
                'files': []
            }
            
            # Add individual file details
            for analysis in class_analyses:
                file_info = {
                    'path': analysis['filepath'],
                    'duration': analysis['duration_seconds'],
                    'sample_rate': analysis['sample_rate'],
                    'channels': analysis['channels'],
                    'file_size_mb': analysis['file_size_mb'],
                    'md5_hash': analysis['md5_hash']
                }
                config_dict['audio']['audio_files'][class_name]['files'].append(file_info)
        
        return config_dict

    def _update_config_with_audio_info(self, analyses: List[Dict], labels: List[int]) -> Dict:
        """Update configuration with audio file information"""
        config_dict = {
            'experiment': vars(self.config.experiment),
            'data': vars(self.config.data),
            'audio': vars(self.config.audio),
            'ga': vars(self.config.ga),
            'model': vars(self.config.model),
            'metrics': self.config.metrics
        }
        
        # Add audio file info - automatically populate the fields
        for analysis, label in zip(analyses, labels):
            config_dict['audio']['audio_files'][f'class_{label}'] = {
                'path': analysis['filepath'],
                'duration': analysis['duration_seconds'],
                'sample_rate': analysis['sample_rate'],
                'channels': analysis['channels'],
                'file_size': analysis['file_size_mb'],
                'md5_hash': analysis['md5_hash']
            }
        
        return config_dict
    
    def _validate_generated_data(self, df: pd.DataFrame, output_dir: str):
        """Final validation of generated data"""
        logger.info("Validating generated dataset...")
        
        # Basic validation
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        
        # Convert ALL numpy types to native Python types
        validation_report = {
            'total_samples': int(len(df)),
            'feature_dimensions': int(len(feature_columns)),
            'classes': [int(x) for x in df['label'].unique().tolist()],
            'class_distribution': {int(k): int(v) for k, v in df['label'].value_counts().to_dict().items()},
            'missing_values': int(df.isnull().sum().sum()),
            'feature_statistics': {
                'mean': float(df[feature_columns].mean().mean()),
                'std': float(df[feature_columns].std().mean()),
                'min': float(df[feature_columns].min().min()),
                'max': float(df[feature_columns].max().max())
            }
        }
        
        # Check for potential issues
        issues = []
        if validation_report['missing_values'] > 0:
            issues.append(f"Dataset contains {validation_report['missing_values']} missing values")
        
        if len(validation_report['classes']) != 2:
            issues.append(f"Expected 2 classes, found {len(validation_report['classes'])}")
        
        class_counts = list(validation_report['class_distribution'].values())
        if len(class_counts) >= 2 and abs(class_counts[0] - class_counts[1]) > min(class_counts) * 0.5:
            issues.append("Significant class imbalance detected")
        
        if validation_report['feature_statistics']['std'] < 0.1:
            issues.append("Low feature variance - data might not be normalized properly")
        
        # Create validation plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Class distribution
        class_dist = validation_report['class_distribution']
        axes[0, 0].bar([str(k) for k in class_dist.keys()], list(class_dist.values()))
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Feature statistics
        feature_means = df[feature_columns].mean()
        axes[0, 1].hist(feature_means, bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution of Feature Means')
        axes[0, 1].set_xlabel('Mean Value')
        axes[0, 1].set_ylabel('Count')
        
        # Sample feature values for first 100 features
        sample_features = df[feature_columns[:100]].values
        im = axes[1, 0].imshow(sample_features[:50].T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Feature Values (First 50 samples, 100 features)')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Feature Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Feature correlation heatmap (sample)
        sample_corr = df[feature_columns[:20]].corr()
        sns.heatmap(sample_corr, ax=axes[1, 1], cmap='coolwarm', center=0)
        axes[1, 1].set_title('Feature Correlation (First 20 features)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_validation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save validation report (now with native Python types)
        with open(f"{output_dir}/validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Log results
        logger.info("Data validation completed:")
        logger.info(f"  Total samples: {validation_report['total_samples']}")
        logger.info(f"  Feature dimensions: {validation_report['feature_dimensions']}")
        logger.info(f"  Class distribution: {validation_report['class_distribution']}")
        
        if issues:
            logger.warning("Data validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ Data validation passed - no issues detected")
        
        logger.info(f"Validation report saved to {output_dir}/validation_report.json")        
        logger.info("Validating generated dataset...")
        
        # Create validation plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Class distribution
        axes[0, 0].bar(validation_report['class_distribution'].keys(), 
                    validation_report['class_distribution'].values())
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Feature statistics
        feature_means = df[feature_columns].mean()
        axes[0, 1].hist(feature_means, bins=30, alpha=0.7)
        axes[0, 1].set_title('Distribution of Feature Means')
        axes[0, 1].set_xlabel('Mean Value')
        axes[0, 1].set_ylabel('Count')
        
        # Sample feature values for first 100 features
        sample_features = df[feature_columns[:100]].values
        im = axes[1, 0].imshow(sample_features[:50].T, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Feature Values (First 50 samples, 100 features)')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].set_ylabel('Feature Index')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Feature correlation heatmap (sample)
        sample_corr = df[feature_columns[:20]].corr()
        sns.heatmap(sample_corr, ax=axes[1, 1], cmap='coolwarm', center=0)
        axes[1, 1].set_title('Feature Correlation (First 20 features)')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_validation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Validating generated dataset before dump...")

        # Save validation report
        with open(f"{output_dir}/validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Log results
        logger.info("Data validation completed:")
        logger.info(f"  Total samples: {validation_report['total_samples']}")
        logger.info(f"  Feature dimensions: {validation_report['feature_dimensions']}")
        logger.info(f"  Class distribution: {validation_report['class_distribution']}")
        
        if issues:
            logger.warning("Data validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ Data validation passed - no issues detected")
        
        logger.info(f"Validation report saved to {output_dir}/validation_report.json")

def generate_audio_data_from_config(config_path: str) -> str:
    """Generate audio data based on configuration with multiple files per class"""
    config = load_config(config_path)
    
    # Parse audio files from config
    class_files_dict = {}
    
    for class_name, class_config in config.audio.audio_files.items():
        if 'paths' in class_config:
            # Multiple files specified as list
            class_files_dict[class_name] = [os.path.expanduser(path) for path in class_config['paths']]
        elif 'path' in class_config:
            # Single file (backward compatibility)
            class_files_dict[class_name] = [os.path.expanduser(class_config['path'])]
        else:
            raise ValueError(f"No 'path' or 'paths' specified for {class_name}")
    
    logger.info(f"Loaded configuration with {len(class_files_dict)} classes:")
    for class_name, file_paths in class_files_dict.items():
        logger.info(f"  {class_name}: {len(file_paths)} files")
    
    generator = AudioDataGenerator(config)
    df = generator.generate_from_files(class_files_dict)
    
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
    
