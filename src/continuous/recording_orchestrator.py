"""
Continuous Recording Orchestrator

Manages continuous recording experiments with automatic QC, training, and monitoring.

Main loop:
1. Start recording
2. Wait for completion
3. Run Auto-QC
4. Process features
5. Train model incrementally
6. Log metrics
7. Repeat until experiment complete
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db_connection import DatabaseConnection


logger = logging.getLogger(__name__)


class ContinuousRecordingOrchestrator:
    """
    Orchestrates continuous recording experiments.

    Manages the complete cycle: record → QC → feature extraction → training
    """

    def __init__(self, experiment_id: str, db_backend: str = 'mariadb'):
        """
        Initialize orchestrator for an experiment.

        Args:
            experiment_id: Unique experiment identifier
            db_backend: 'mariadb' or 'sqlite'
        """
        self.experiment_id = experiment_id
        self.db = DatabaseConnection(backend=db_backend)
        self.logger = logging.getLogger(f"{__name__}.{experiment_id}")

        # Will be loaded from database
        self.config = None
        self.current_cycle = 0
        self.should_stop = False

        # Auto-QC validator (will be initialized with config)
        self.qc_validator = None

        # Track last recording file path from API
        self._last_recording_path = None

    def load_experiment_config(self) -> Dict[str, Any]:
        """Load experiment configuration from database"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)
            cursor.execute("""
                SELECT * FROM continuous_experiments
                WHERE experiment_id = %s
            """, (self.experiment_id,))

            config = cursor.fetchone()
            if not config:
                raise ValueError(f"Experiment {self.experiment_id} not found")

            # Resolve substance names to class labels
            from continuous.substance_vocabulary import get_class_for_substance

            config['channel_1_expected_class'] = get_class_for_substance(config['channel_1_substance'])
            config['channel_2_expected_class'] = get_class_for_substance(config['channel_2_substance'])

            if config['channel_1_expected_class'] is None:
                raise ValueError(f"Invalid substance for channel 1: {config['channel_1_substance']}")
            if config['channel_2_expected_class'] is None:
                raise ValueError(f"Invalid substance for channel 2: {config['channel_2_substance']}")

            # Keep substance names for logging/metadata
            config['channel_1_source'] = config['channel_1_substance']
            config['channel_2_source'] = config['channel_2_substance']

            self.config = config
            self.current_cycle = config['current_cycle']
            return config

    def update_experiment_status(self, status: str, **kwargs):
        """Update experiment status in database"""
        updates = ['status = %s', 'updated_at = NOW()']
        values = [status]

        for key, value in kwargs.items():
            updates.append(f"{key} = %s")
            values.append(value)

        values.append(self.experiment_id)

        with self.db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)
            cursor.execute(f"""
                UPDATE continuous_experiments
                SET {', '.join(updates)}
                WHERE experiment_id = %s
            """, values)
            conn.commit()

    def create_recording_cycle(self, cycle_number: int) -> int:
        """Create new recording cycle entry"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)
            cursor.execute("""
                INSERT INTO recording_cycles
                (experiment_id, cycle_number, status, start_time)
                VALUES (%s, %s, 'scheduled', NOW())
            """, (self.experiment_id, cycle_number))
            conn.commit()
            return cursor.lastrowid

    def update_cycle_status(self, cycle_number: int, status: str, **kwargs):
        """Update recording cycle status"""
        updates = ['status = %s', 'updated_at = NOW()']
        values = [status]

        for key, value in kwargs.items():
            updates.append(f"{key} = %s")
            values.append(value)

        values.extend([self.experiment_id, cycle_number])

        with self.db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)
            cursor.execute(f"""
                UPDATE recording_cycles
                SET {', '.join(updates)}
                WHERE experiment_id = %s AND cycle_number = %s
            """, values)
            conn.commit()

    def log_alert(self, alert_type: str, message: str, severity: str = 'info',
                  cycle_number: Optional[int] = None, details: Optional[Dict] = None):
        """Log alert to database"""
        import json

        with self.db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)
            cursor.execute("""
                INSERT INTO experiment_alerts
                (experiment_id, cycle_number, alert_type, severity, message, details)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (self.experiment_id, cycle_number, alert_type, severity, message,
                  json.dumps(details) if details else None))
            conn.commit()

    async def start_recording_cycle(self, cycle_number: int) -> Optional[str]:
        """
        Start a new recording cycle.

        Returns:
            session_id if successful, None if failed
        """
        self.logger.info(f"Starting cycle {cycle_number}")

        try:
            # Import here to avoid circular dependencies
            import requests
            import os

            # Get rubix44 URL
            rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')

            # Prepare recording parameters
            params = {
                'playback_file': self.config['playback_file'],
                'duration': self.config['recording_duration_seconds'],
                'output_prefix': f"{self.config['output_prefix']}_exp{self.experiment_id}_cycle{cycle_number}"
            }

            # Start recording
            response = requests.post(
                f"{rubix_url}/api/v1/recordings/start",
                json=params,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()

            # API returns session info nested under 'session' key
            # The session ID is in session['id']
            session_data = result.get('session', {})
            session_id = session_data.get('id')

            # Fallback: try direct session_id key for backward compatibility
            if not session_id:
                session_id = result.get('session_id')

            if not session_id:
                self.logger.error(f"API response: {result}")
                raise ValueError("No session_id returned from rubix44")

            # Extract additional session metadata (added Jan 2026)
            human_id = session_data.get('human_id', 'N/A')
            status = session_data.get('status', 'unknown')
            duration = session_data.get('duration', self.config['recording_duration_seconds'])

            self.logger.info(f"Recording started: {session_id} ({human_id})")
            self.logger.info(f"  Status: {status}, Duration: {duration}s")

            # NOTE: Don't update cycle with session_id yet - recording_sessions entry
            # must be created first due to foreign key constraint.
            # Session ID will be set in create_recording_metadata()
            self.update_cycle_status(cycle_number, 'recording')

            return session_id

        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.log_alert('rubix44_error', f"Failed to start recording: {str(e)}",
                          severity='error', cycle_number=cycle_number)
            self.update_cycle_status(cycle_number, 'failed', error_message=str(e))
            return None

    async def wait_for_recording_completion(self, session_id: str, cycle_number: int) -> bool:
        """
        Wait for recording to complete.

        Polls rubix44 status every 30 seconds.

        Returns:
            True if completed successfully, False if error
        """
        import requests
        import os

        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')
        max_wait_hours = 2  # Safety timeout
        start_time = time.time()

        self.logger.info(f"Waiting for recording {session_id} to complete...")

        while True:
            try:
                # Check if we should stop
                if self.should_stop:
                    self.logger.info("Stop requested, aborting wait")
                    return False

                # Safety timeout
                if (time.time() - start_time) / 3600 > max_wait_hours:
                    self.logger.error(f"Recording timeout after {max_wait_hours} hours")
                    self.log_alert('rubix44_error', f"Recording timeout after {max_wait_hours} hours",
                                  severity='critical', cycle_number=cycle_number)
                    return False

                # Poll status
                response = requests.get(f"{rubix_url}/api/v1/recordings/status", timeout=30)
                response.raise_for_status()
                status_data = response.json()

                # Handle both nested 'session' format and direct format
                if 'session' in status_data:
                    session_info = status_data['session']
                    current_status = session_info.get('status', 'unknown')
                else:
                    session_info = status_data
                    current_status = status_data.get('status', 'unknown')

                # Extract progress information
                progress_percent = session_info.get('progress_percent', 0)
                elapsed_seconds = session_info.get('elapsed_seconds', 0)
                error_msg = session_info.get('error')

                # Handle different status values
                if current_status == 'completed':
                    # Recording completed successfully - extract file paths
                    files = session_info.get('files', [])
                    stereo_file_path = None
                    if files:
                        # Find the stereo WAV file
                        for file_info in files:
                            if '_stereo.wav' in file_info.get('path', ''):
                                stereo_file_path = file_info['path']
                                break

                    if stereo_file_path:
                        self.logger.info(f"Recording {session_id} completed successfully")
                        self.logger.info(f"  Stereo file: {stereo_file_path}")
                        # Store file path for later use
                        self._last_recording_path = stereo_file_path
                    else:
                        self.logger.warning(f"Recording {session_id} completed but no stereo file path found")

                    self.update_cycle_status(cycle_number, 'qc_pending',
                                            end_time=datetime.now())
                    return True
                elif current_status == 'idle':
                    # Recording complete (legacy status)
                    self.logger.info(f"Recording {session_id} completed")
                    self.update_cycle_status(cycle_number, 'qc_pending',
                                            end_time=datetime.now())
                    return True
                elif current_status == 'error':
                    # Recording failed
                    error = error_msg or 'Unknown error'
                    self.logger.error(f"Recording {session_id} failed: {error}")
                    self.log_alert('rubix44_error', f"Recording failed: {error}",
                                  severity='critical', cycle_number=cycle_number)
                    return False
                elif current_status == 'stopped':
                    # Recording was stopped manually
                    self.logger.warning(f"Recording {session_id} was stopped manually")
                    self.update_cycle_status(cycle_number, 'failed',
                                            error_message='Recording stopped manually')
                    return False
                elif current_status == 'recording':
                    # Still recording - log progress
                    self.logger.info(f"Recording {session_id}: {progress_percent:.1f}% complete ({elapsed_seconds:.0f}s elapsed)")
                    await asyncio.sleep(30)  # Poll every 30 seconds
                else:
                    self.logger.warning(f"Unknown status: {current_status}")
                    await asyncio.sleep(30)

            except Exception as e:
                self.logger.error(f"Error checking recording status: {e}")
                await asyncio.sleep(30)  # Wait and retry

    def get_recording_filename(self, session_id: str) -> Optional[str]:
        """
        Get the actual stereo WAV filename for a session by querying history.

        Args:
            session_id: Session ID returned from start recording

        Returns:
            Stereo WAV filename or None if not found
        """
        import requests
        import os

        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')

        try:
            response = requests.get(f"{rubix_url}/api/v1/recordings/history", timeout=30)
            response.raise_for_status()
            history = response.json()

            # Find session in history
            for session in history:
                if session['id'] == session_id:
                    # Find stereo file
                    for file_info in session.get('files', []):
                        if '_stereo.wav' in file_info.get('name', ''):
                            self.logger.info(f"Found stereo file for {session_id}: {file_info['name']}")
                            return file_info['name']

            self.logger.error(f"No stereo file found for session {session_id}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to get filename for {session_id}: {e}")
            return None

    async def download_recording(self, filename: str, local_path: Path) -> bool:
        """
        Download recording file from rubix44 server.

        Args:
            filename: Name of the file to download (e.g., "prefix_2026-01-17_12-34-56_stereo.wav")
            local_path: Local path where file should be saved

        Returns:
            bool: True if download successful, False otherwise
        """
        import requests
        import os

        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')
        download_url = f"{rubix_url}/api/v1/recordings/{filename}"

        try:
            self.logger.info(f"Downloading from: {download_url}")
            response = requests.get(download_url, stream=True, timeout=120)
            response.raise_for_status()

            # Download in chunks
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify file was downloaded
            if not local_path.exists():
                self.logger.error(f"Download completed but file not found: {local_path}")
                return False

            file_size = local_path.stat().st_size
            if file_size == 0:
                self.logger.error(f"Downloaded file is empty: {local_path}")
                local_path.unlink()  # Delete empty file
                return False

            self.logger.info(f"Successfully downloaded {filename} ({file_size:,} bytes)")
            return True

        except requests.exceptions.Timeout:
            self.logger.error(f"Download timeout for {filename}")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Download failed for {filename}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {filename}: {e}")
            if local_path.exists():
                local_path.unlink()  # Clean up partial download
            return False

    async def create_recording_metadata(self, session_id: str, cycle_number: int):
        """
        Create metadata entry for recording in recording_sessions table.

        Auto-fills from experiment configuration.
        """
        # TODO: Implement metadata creation
        # For now, this is a placeholder
        self.logger.info(f"Creating metadata for {session_id}")

        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor(buffered=True)

                # Check if metadata already exists
                cursor.execute("""
                    SELECT session_id FROM recording_sessions
                    WHERE session_id = %s
                """, (session_id,))

                if cursor.fetchone():
                    self.logger.info("Metadata already exists, updating...")
                    cursor.execute("""
                        UPDATE recording_sessions
                        SET experiment_id = %s,
                            cycle_number = %s,
                            channel_1_source = %s,
                            channel_1_expected_class = %s,
                            channel_2_source = %s,
                            channel_2_expected_class = %s,
                            beaker_1_role = %s,
                            beaker_1_content = %s,
                            beaker_2_role = %s,
                            beaker_2_content = %s,
                            faraday_cage_used = %s,
                            researcher_name = %s
                        WHERE session_id = %s
                    """, (
                        self.experiment_id,
                        cycle_number,
                        self.config['channel_1_source'],
                        self.config['channel_1_expected_class'],
                        self.config['channel_2_source'],
                        self.config['channel_2_expected_class'],
                        self.config['beaker_1_role'],
                        self.config['beaker_1_content'],
                        self.config['beaker_2_role'],
                        self.config['beaker_2_content'],
                        self.config['faraday_cage_used'],
                        self.config['researcher_name'],
                        session_id
                    ))
                else:
                    self.logger.info("Creating new metadata entry...")
                    cursor.execute("""
                        INSERT INTO recording_sessions
                        (session_id, experiment_id, cycle_number, recording_date,
                         channel_1_source, channel_1_expected_class,
                         channel_2_source, channel_2_expected_class,
                         beaker_1_role, beaker_1_content,
                         beaker_2_role, beaker_2_content,
                         beaker_3_role, beaker_3_content,
                         faraday_cage_used, researcher_name)
                        VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, 'not_used', '', %s, %s)
                    """, (
                        session_id,
                        self.experiment_id,
                        cycle_number,
                        self.config['channel_1_source'],
                        self.config['channel_1_expected_class'],
                        self.config['channel_2_source'],
                        self.config['channel_2_expected_class'],
                        self.config['beaker_1_role'],
                        self.config['beaker_1_content'],
                        self.config['beaker_2_role'],
                        self.config['beaker_2_content'],
                        self.config['faraday_cage_used'],
                        self.config['researcher_name']
                    ))

                conn.commit()
                self.logger.info("Metadata created successfully")

                # Now that recording_sessions entry exists, we can safely update
                # recording_cycles with the session_id (satisfies foreign key constraint)
                # Note: We don't change the status here, just add the session_id
                cursor.execute("""
                    UPDATE recording_cycles
                    SET session_id = %s
                    WHERE experiment_id = %s AND cycle_number = %s
                """, (session_id, self.experiment_id, cycle_number))
                conn.commit()
                self.logger.debug(f"Updated cycle {cycle_number} with session_id {session_id}")

        except Exception as e:
            self.logger.error(f"Failed to create metadata: {e}")
            raise

    async def run_cycle(self, cycle_number: int) -> bool:
        """
        Run a single recording cycle.

        Returns:
            True if successful, False if failed
        """
        import os

        self.logger.info(f"=" * 60)
        self.logger.info(f"CYCLE {cycle_number} START")
        self.logger.info(f"=" * 60)

        # Create cycle entry
        cycle_id = self.create_recording_cycle(cycle_number)

        try:
            # Step 1: Start recording
            session_id = await self.start_recording_cycle(cycle_number)
            if not session_id:
                return False

            # Step 2: Wait for completion
            success = await self.wait_for_recording_completion(session_id, cycle_number)
            if not success:
                return False

            # Step 3: Create metadata
            await self.create_recording_metadata(session_id, cycle_number)

            # Step 4: Download recording file from rubix44 server
            # Get actual filename from history (format: prefix_timestamp_stereo.wav)
            stereo_filename = self.get_recording_filename(session_id)
            if not stereo_filename:
                self.logger.error(f"Could not find stereo file for session {session_id}")
                self.update_cycle_status(cycle_number, 'failed',
                                        error_message=f"Stereo file not found for {session_id}")
                return False

            local_recordings_dir = Path(os.getenv('RUBIX44_LOCAL_DIR', '/Users/bernd/rubix44/recordings'))
            local_recordings_dir.mkdir(parents=True, exist_ok=True)
            wav_path = local_recordings_dir / stereo_filename

            # Download from rubix44 server if not already present
            if not wav_path.exists():
                self.logger.info(f"Downloading {stereo_filename} from rubix44 server...")
                success = await self.download_recording(stereo_filename, wav_path)
                if not success:
                    self.logger.error(f"Failed to download recording {stereo_filename}")
                    self.update_cycle_status(cycle_number, 'failed',
                                            error_message=f"Failed to download {stereo_filename}")
                    return False
                self.logger.info(f"Downloaded {stereo_filename} ({wav_path.stat().st_size} bytes)")
            else:
                self.logger.info(f"Using cached file: {wav_path}")

            # Step 5: Run Auto-QC
            if not self.qc_validator:
                from auto_qc_validator import AutoQCValidator
                self.qc_validator = AutoQCValidator(self.config)

            self.logger.info(f"Running Auto-QC on {wav_path}")
            qc_result = await self.qc_validator.validate_recording(
                session_id=session_id,
                wav_path=wav_path,
                duration_seconds=self.config['recording_duration_seconds'],
                channel_1_class=self.config['channel_1_expected_class'],
                channel_2_class=self.config['channel_2_expected_class']
            )

            # Update database with QC results
            self.update_cycle_status(
                cycle_number,
                'qc_completed' if qc_result.passed else 'qc_failed',
                qc_passed=qc_result.passed,
                qc_separation_score=qc_result.separation_score,
                qc_silhouette_score=qc_result.silhouette_score,
                qc_decision=qc_result.decision,
                qc_notes=qc_result.notes,
                samples_extracted_ch1=qc_result.samples_ch1,
                samples_extracted_ch2=qc_result.samples_ch2
            )

            # If QC failed, skip remaining steps
            if not qc_result.passed:
                self.logger.warning(f"QC failed: {qc_result.notes}")
                self.log_alert('qc_failure', qc_result.notes, 'warning', cycle_number)
                return False

            self.logger.info(f"QC passed! Separation score: {qc_result.separation_score:.3f}")

            # Step 5: Process features (if QC passed)
            from stereo_channel_processor import StereoChannelProcessor

            # Create config object for processor
            from types import SimpleNamespace
            audio_config = SimpleNamespace(
                sample_rate=22050,  # Default sample rate
                segment_duration=1.0,  # 1 second segments
                n_mfcc=13,  # Default MFCC coefficients
                feature_types=['mfcc', 'spectral', 'chroma', 'tonnetz']
            )
            processor_config = SimpleNamespace(audio=audio_config)
            processor = StereoChannelProcessor(processor_config)

            self.logger.info("Extracting features from full recording...")
            self.update_cycle_status(cycle_number, 'processing')

            try:
                X_left, y_left, X_right, y_right, metadata = processor.process_stereo_file(
                    wav_file_path=str(wav_path),
                    positive_label=self.config['channel_1_expected_class'],
                    negative_label=self.config['channel_2_expected_class'],
                    samples_per_channel=None  # Extract all samples
                )

                # Store features in database
                from feature_database import FeatureDatabase
                feature_db = FeatureDatabase(backend='mariadb')

                # Combine features
                import numpy as np
                X_combined = np.vstack([X_left, X_right])
                y_combined = np.concatenate([y_left, y_right])

                # Store in database
                feature_db.store_features(
                    session_id=session_id,
                    features=X_combined,
                    labels=y_combined,
                    metadata={
                        'experiment_id': self.experiment_id,
                        'cycle_number': cycle_number,
                        'channel_1_samples': len(X_left),
                        'channel_2_samples': len(X_right),
                        'qc_separation_score': qc_result.separation_score
                    }
                )

                self.logger.info(f"Stored {len(X_combined)} features in database")
                self.update_cycle_status(cycle_number, 'features_extracted',
                                        features_extracted=True,
                                        total_samples=len(X_combined))

            except Exception as e:
                self.logger.error(f"Feature extraction failed: {e}")
                self.log_alert('feature_extraction_error', str(e), 'error', cycle_number)
                return False

            # Step 6: Train model incrementally
            from incremental_trainer import IncrementalTrainer
            trainer = IncrementalTrainer(
                model_dir=Path(f"models/continuous/{self.experiment_id}"),
                db_backend='mariadb'
            )

            self.logger.info("Starting incremental training...")
            self.update_cycle_status(cycle_number, 'training')

            try:
                # Train on sliding window (last N weeks)
                training_window_weeks = self.config.get('training_sliding_window_weeks', 2)
                training_result = trainer.train_incremental(
                    experiment_id=self.experiment_id,
                    training_window_weeks=training_window_weeks,
                    batch_size=self.config.get('training_batch_size', 32),
                    epochs=self.config.get('training_epochs_per_cycle', 5)
                )

                # Update cycle with training results
                self.update_cycle_status(
                    cycle_number,
                    'completed',
                    model_accuracy=training_result.get('accuracy', 0.0),
                    training_time_seconds=training_result.get('training_time', 0.0)
                )

                # Update experiment current accuracy
                self.update_experiment_status('running',
                                             current_accuracy=training_result.get('accuracy', 0.0))

                self.logger.info(f"Training complete! Accuracy: {training_result.get('accuracy', 0.0):.3f}")

            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                self.log_alert('training_error', str(e), 'error', cycle_number)
                self.update_cycle_status(cycle_number, 'failed', error_message=str(e))
                return False

            self.logger.info(f"Cycle {cycle_number} completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Cycle {cycle_number} failed: {e}")
            self.update_cycle_status(cycle_number, 'failed', error_message=str(e))
            self.log_alert('other', f"Cycle {cycle_number} failed: {str(e)}",
                          severity='error', cycle_number=cycle_number)
            return False

    async def run_experiment(self):
        """
        Main experiment loop.

        Runs until target duration is reached or manually stopped.
        """
        # Load configuration
        self.load_experiment_config()

        self.logger.info(f"Starting experiment: {self.config['experiment_name']}")
        self.logger.info(f"Target duration: {self.config['target_duration_weeks']} weeks")
        self.logger.info(f"Recording interval: {self.config['recording_interval_minutes']} minutes")

        # Calculate total cycles expected (supports fractional weeks)
        target_weeks = float(self.config['target_duration_weeks'])
        total_minutes = target_weeks * 7 * 24 * 60
        total_cycles = int(total_minutes / self.config['recording_interval_minutes'])

        self.logger.info(f"Calculated total cycles: {total_cycles} ({total_minutes:.1f} minutes total)")

        self.update_experiment_status('running', total_cycles_expected=total_cycles)

        # Main loop
        while self.current_cycle < total_cycles:
            # Check database for pause request
            self._check_pause_request()

            if self.should_stop:
                self.logger.info("Stop requested, pausing experiment")
                self.update_experiment_status('paused', orchestrator_pid=None)
                break

            self.current_cycle += 1

            # Run cycle
            success = await self.run_cycle(self.current_cycle)

            # Update experiment metrics
            if success:
                self.update_experiment_status('running',
                                             current_cycle=self.current_cycle,
                                             qc_pass_count=self.config['qc_pass_count'] + 1)

            # Sleep until next cycle (if not the last one)
            if self.current_cycle < total_cycles:
                sleep_seconds = self.config['recording_interval_minutes'] * 60
                next_cycle_time = datetime.now() + timedelta(seconds=sleep_seconds)

                self.update_experiment_status('running', next_cycle_scheduled=next_cycle_time)

                self.logger.info(f"Sleeping {sleep_seconds}s until next cycle...")
                self.logger.info(f"Next cycle at: {next_cycle_time}")

                await asyncio.sleep(sleep_seconds)

        # Experiment complete
        if self.current_cycle >= total_cycles:
            self.logger.info("Experiment completed!")
            self.update_experiment_status('completed', completed_at=datetime.now(), orchestrator_pid=None)
            self.log_alert('completion',
                          f"Experiment {self.config['experiment_name']} completed successfully!",
                          severity='info')

    def _check_pause_request(self):
        """Check database for pause request from user"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)
            cursor.execute("""
                SELECT status FROM continuous_experiments
                WHERE experiment_id = %s
            """, (self.experiment_id,))

            result = cursor.fetchone()
            if result and result['status'] == 'paused':
                self.logger.info("Pause detected from database")
                self.should_stop = True

    def stop(self):
        """Request orchestrator to stop gracefully"""
        self.logger.info("Stop requested")
        self.should_stop = True


def main():
    """CLI entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Run continuous recording experiment')
    parser.add_argument('experiment_id', help='Experiment ID to run')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run orchestrator
    orchestrator = ContinuousRecordingOrchestrator(args.experiment_id)

    try:
        asyncio.run(orchestrator.run_experiment())
    except KeyboardInterrupt:
        print("\n⚠ Interrupted! Stopping gracefully...")
        orchestrator.stop()


if __name__ == '__main__':
    main()
