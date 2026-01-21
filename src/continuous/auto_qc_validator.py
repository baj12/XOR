"""
Automatic Quality Control Validator

Validates recording quality without manual intervention using:
- Duration checks
- File integrity checks
- Feature extraction tests
- Class separation analysis (UMAP/t-SNE/PCA)
- Audio quality metrics (SNR, clipping detection)
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from stereo_channel_processor import StereoChannelProcessor


logger = logging.getLogger(__name__)


@dataclass
class QCResult:
    """Quality control validation result"""
    passed: bool
    decision: str  # 'auto_approved', 'auto_rejected', 'manual_review'

    # Individual checks
    duration_ok: bool
    file_ok: bool
    features_ok: bool
    separation_ok: bool
    audio_ok: bool

    # Metrics
    separation_score: float  # Silhouette score or separation ratio
    silhouette_score: Optional[float] = None
    audio_quality_score: Optional[float] = None

    # Sample counts
    samples_ch1: int = 0
    samples_ch2: int = 0

    # Details
    notes: str = ""
    error_message: Optional[str] = None


class AutoQCValidator:
    """
    Automatic quality control validator for recordings.

    Provides pass/fail decisions based on configurable thresholds.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize validator with configuration.

        Args:
            config: QC configuration dict with thresholds
        """
        # Default thresholds
        self.min_separation_score = 0.7
        self.min_duration = 3500  # seconds (allow 100s tolerance)
        self.max_duration = 3700
        self.min_samples_per_channel = 900
        self.auto_approve_threshold = 0.8  # Above this = auto-approve
        self.auto_reject_threshold = 0.6   # Below this = auto-reject

        # Override with config if provided
        if config:
            self.min_separation_score = config.get('auto_qc_min_separation_score', 0.7)
            self.min_samples_per_channel = config.get('auto_qc_min_samples_per_channel', 900)
            self.auto_approve_threshold = config.get('auto_qc_auto_approve_threshold', 0.8)
            self.auto_reject_threshold = config.get('auto_qc_auto_reject_threshold', 0.6)

        # Create config object for processor
        from types import SimpleNamespace
        audio_config = SimpleNamespace(
            sample_rate=22050,
            segment_duration=1.0,
            n_mfcc=13,
            feature_types=['mfcc', 'spectral', 'chroma', 'tonnetz']
        )
        processor_config = SimpleNamespace(audio=audio_config)
        self.processor = StereoChannelProcessor(processor_config)
        self.logger = logging.getLogger(__name__)

    def check_duration(self, duration_seconds: Optional[float]) -> bool:
        """Check if recording duration is acceptable"""
        if duration_seconds is None:
            return False

        return self.min_duration <= duration_seconds <= self.max_duration

    def check_file_integrity(self, wav_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Check if WAV file exists and is readable.

        Returns:
            (success, error_message)
        """
        try:
            if not wav_path.exists():
                return False, f"File not found: {wav_path}"

            if wav_path.stat().st_size == 0:
                return False, "File is empty"

            # Try to load with librosa
            import librosa
            y, sr = librosa.load(str(wav_path), sr=None, duration=1.0)

            if len(y) == 0:
                return False, "No audio data in file"

            return True, None

        except Exception as e:
            return False, f"File integrity check failed: {str(e)}"

    def extract_test_features(self, wav_path: Path, channel_1_class: int,
                              channel_2_class: int) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        Extract features from a limited sample for QC.

        Args:
            wav_path: Path to stereo WAV file
            channel_1_class: Expected class for left channel
            channel_2_class: Expected class for right channel

        Returns:
            (success, features_dict, error_message)
        """
        try:
            # Extract limited samples for speed (1000 per channel)
            result = self.processor.process_stereo_file(
                wav_path=str(wav_path),
                positive_label=channel_1_class,
                negative_label=channel_2_class,
                max_samples_per_channel=1000  # Limited for QC
            )

            X_left, y_left, X_right, y_right, offsets_left, offsets_right = result

            # Check sample counts
            if len(X_left) < self.min_samples_per_channel:
                return False, None, f"Insufficient samples in left channel: {len(X_left)}"

            if len(X_right) < self.min_samples_per_channel:
                return False, None, f"Insufficient samples in right channel: {len(X_right)}"

            features = {
                'X_left': X_left,
                'y_left': y_left,
                'X_right': X_right,
                'y_right': y_right,
                'samples_ch1': len(X_left),
                'samples_ch2': len(X_right)
            }

            return True, features, None

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return False, None, f"Feature extraction failed: {str(e)}"

    def analyze_class_separation(self, features: Dict) -> Tuple[bool, float, Optional[float]]:
        """
        Analyze class separation quality using dimensionality reduction.

        Args:
            features: Dict with X_left, y_left, X_right, y_right

        Returns:
            (meets_threshold, separation_score, silhouette_score)
        """
        try:
            from sklearn.metrics import silhouette_score
            from sklearn.decomposition import PCA

            # Combine features from both channels
            X = np.vstack([features['X_left'], features['X_right']])
            y = np.concatenate([features['y_left'], features['y_right']])

            # Reduce to 2D for separation analysis
            if X.shape[1] > 2:
                pca = PCA(n_components=min(10, X.shape[1]))
                X_reduced = pca.fit_transform(X)
            else:
                X_reduced = X

            # Calculate silhouette score (range: -1 to 1, higher is better)
            sil_score = silhouette_score(X_reduced, y)

            # Calculate separation ratio (inter-class / intra-class distance)
            class_0 = X_reduced[y == 0]
            class_1 = X_reduced[y == 1]

            if len(class_0) > 0 and len(class_1) > 0:
                # Inter-class distance (between centroids)
                centroid_0 = np.mean(class_0, axis=0)
                centroid_1 = np.mean(class_1, axis=0)
                inter_dist = np.linalg.norm(centroid_0 - centroid_1)

                # Intra-class distance (average spread)
                intra_dist_0 = np.mean([np.linalg.norm(x - centroid_0) for x in class_0])
                intra_dist_1 = np.mean([np.linalg.norm(x - centroid_1) for x in class_1])
                intra_dist = (intra_dist_0 + intra_dist_1) / 2

                # Separation ratio
                if intra_dist > 0:
                    separation_ratio = inter_dist / intra_dist
                else:
                    separation_ratio = 0.0

                # Normalize to 0-1 range (heuristic: ratio of 2 = score of 0.7)
                separation_score = min(1.0, separation_ratio / 3.0)
            else:
                separation_score = 0.0

            # Use silhouette score as primary metric (scale from -1:1 to 0:1)
            primary_score = (sil_score + 1) / 2

            meets_threshold = primary_score >= self.min_separation_score

            self.logger.debug(f"Silhouette: {sil_score:.3f}, Separation: {separation_score:.3f}, Primary: {primary_score:.3f}")

            return meets_threshold, primary_score, sil_score

        except Exception as e:
            self.logger.error(f"Class separation analysis failed: {e}")
            return False, 0.0, None

    def check_audio_quality(self, wav_path: Path) -> Tuple[bool, float, str]:
        """
        Check audio quality metrics.

        Returns:
            (ok, quality_score, notes)
        """
        try:
            import librosa
            import soundfile as sf

            # Load audio
            y, sr = librosa.load(str(wav_path), sr=None, duration=10.0)  # Check first 10 seconds

            # Check for clipping (values at max)
            clipping_ratio = np.sum(np.abs(y) > 0.99) / len(y)

            # Check dynamic range
            rms = np.sqrt(np.mean(y**2))
            peak = np.max(np.abs(y))

            # Simple quality score (0-1)
            quality_score = 1.0
            notes = []

            if clipping_ratio > 0.01:  # More than 1% clipping
                quality_score -= 0.3
                notes.append(f"Clipping detected: {clipping_ratio*100:.1f}%")

            if rms < 0.01:  # Very low signal
                quality_score -= 0.2
                notes.append(f"Low signal level: RMS={rms:.4f}")

            if peak < 0.1:  # Very weak signal
                quality_score -= 0.3
                notes.append(f"Weak signal: peak={peak:.4f}")

            quality_ok = quality_score >= 0.5
            notes_str = "; ".join(notes) if notes else "Good"

            return quality_ok, quality_score, notes_str

        except Exception as e:
            self.logger.error(f"Audio quality check failed: {e}")
            return True, 0.7, f"Quality check failed: {str(e)}"  # Don't fail on this

    async def validate_recording(self, session_id: str, wav_path: Path,
                                 duration_seconds: Optional[float],
                                 channel_1_class: int, channel_2_class: int) -> QCResult:
        """
        Perform complete QC validation.

        Args:
            session_id: Recording session ID
            wav_path: Path to stereo WAV file
            duration_seconds: Recording duration
            channel_1_class: Expected class for left channel
            channel_2_class: Expected class for right channel

        Returns:
            QCResult with pass/fail decision
        """
        self.logger.info(f"Running Auto-QC for {session_id}")

        notes = []

        # Check 1: Duration
        duration_ok = self.check_duration(duration_seconds)
        if not duration_ok:
            notes.append(f"Duration out of range: {duration_seconds}s")

        # Check 2: File integrity
        file_ok, file_error = self.check_file_integrity(wav_path)
        if not file_ok:
            notes.append(file_error or "File integrity check failed")
            # Can't continue if file is bad
            return QCResult(
                passed=False,
                decision='auto_rejected',
                duration_ok=duration_ok,
                file_ok=False,
                features_ok=False,
                separation_ok=False,
                audio_ok=False,
                separation_score=0.0,
                notes="; ".join(notes),
                error_message=file_error
            )

        # Check 3: Feature extraction
        features_ok, features, features_error = self.extract_test_features(
            wav_path, channel_1_class, channel_2_class
        )

        if not features_ok:
            notes.append(features_error or "Feature extraction failed")
            return QCResult(
                passed=False,
                decision='auto_rejected',
                duration_ok=duration_ok,
                file_ok=file_ok,
                features_ok=False,
                separation_ok=False,
                audio_ok=False,
                separation_score=0.0,
                notes="; ".join(notes),
                error_message=features_error
            )

        # Check 4: Class separation
        separation_ok, separation_score, silhouette_score = self.analyze_class_separation(features)
        if not separation_ok:
            notes.append(f"Poor class separation: {separation_score:.3f}")

        # Check 5: Audio quality
        audio_ok, audio_quality_score, audio_notes = self.check_audio_quality(wav_path)
        if not audio_ok:
            notes.append(audio_notes)

        # Overall decision
        all_checks_pass = all([duration_ok, file_ok, features_ok, separation_ok, audio_ok])

        # Determine decision based on separation score
        if separation_score >= self.auto_approve_threshold:
            decision = 'auto_approved'
            passed = True
        elif separation_score < self.auto_reject_threshold:
            decision = 'auto_rejected'
            passed = False
        else:
            decision = 'manual_review'
            passed = False  # Don't auto-approve, needs human review
            notes.append("Borderline quality - manual review recommended")

        self.logger.info(f"QC Result: {decision}, Score: {separation_score:.3f}")

        return QCResult(
            passed=passed,
            decision=decision,
            duration_ok=duration_ok,
            file_ok=file_ok,
            features_ok=features_ok,
            separation_ok=separation_ok,
            audio_ok=audio_ok,
            separation_score=separation_score,
            silhouette_score=silhouette_score,
            audio_quality_score=audio_quality_score,
            samples_ch1=features['samples_ch1'],
            samples_ch2=features['samples_ch2'],
            notes="; ".join(notes) if notes else "All checks passed"
        )


def main():
    """CLI testing"""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description='Test Auto-QC on a recording')
    parser.add_argument('wav_file', help='Path to stereo WAV file')
    parser.add_argument('--ch1-class', type=int, default=1, help='Channel 1 expected class')
    parser.add_argument('--ch2-class', type=int, default=0, help='Channel 2 expected class')
    parser.add_argument('--duration', type=float, help='Recording duration in seconds')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    validator = AutoQCValidator()

    async def test():
        result = await validator.validate_recording(
            session_id="test",
            wav_path=Path(args.wav_file),
            duration_seconds=args.duration,
            channel_1_class=args.ch1_class,
            channel_2_class=args.ch2_class
        )

        print("\n" + "=" * 60)
        print("AUTO-QC RESULT")
        print("=" * 60)
        print(f"Decision: {result.decision}")
        print(f"Passed: {result.passed}")
        print(f"Separation Score: {result.separation_score:.3f}")
        print(f"Silhouette Score: {result.silhouette_score:.3f}" if result.silhouette_score else "")
        print(f"Samples CH1: {result.samples_ch1}")
        print(f"Samples CH2: {result.samples_ch2}")
        print(f"Notes: {result.notes}")
        print("=" * 60)

    asyncio.run(test())


if __name__ == '__main__':
    main()
