"""
Rubix44 Data Provider for Continuous Learning

Connects to rubix44-recorder API to automatically fetch and process new stereo recordings
for the continuous learning pipeline.
"""

import requests
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
import json
import hashlib
import sys

# Import database connection
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
from db_connection import DatabaseConnection

logger = logging.getLogger(__name__)


class Rubix44Client:
    """
    Client for rubix44-recorder API.

    Communicates with the remote recorder to fetch recording metadata and files.
    """

    def __init__(self, base_url: str = "http://10.0.0.58:5000"):
        """
        Initialize rubix44 API client.

        Args:
            base_url: Base URL of the rubix44-recorder API (default: http://10.0.0.58:5000)
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/v1"
        logger.info(f"Rubix44Client initialized: {self.api_base}")

    def health_check(self) -> bool:
        """
        Check if API server is healthy using dedicated health endpoint.

        Returns:
            True if server is accessible and reports healthy status
        """
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                is_healthy = data.get('status') == 'healthy'
                if is_healthy:
                    logger.debug(f"Health check passed: {data.get('service', 'unknown')} at {data.get('timestamp')}")
                return is_healthy
            return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_config(self) -> Dict:
        """
        Get current recorder configuration.

        Returns:
            Configuration dictionary
        """
        response = requests.get(f"{self.api_base}/config", timeout=5)
        response.raise_for_status()
        return response.json()

    def get_recording_status(self) -> Dict:
        """
        Get current recording status.

        Returns:
            Status dictionary with 'status' and optional session info
        """
        response = requests.get(f"{self.api_base}/recordings/status", timeout=5)
        response.raise_for_status()
        return response.json()

    def get_devices(self) -> List[Dict]:
        """
        Get list of all available audio devices.

        Returns:
            List of device dictionaries with id, name, channels, sample_rate
        """
        response = requests.get(f"{self.api_base}/devices", timeout=5)
        response.raise_for_status()
        return response.json()

    def get_rubix_device(self) -> Optional[Dict]:
        """
        Get Rubix44 device information specifically.

        Returns:
            Device info dict if Rubix44 is found, None otherwise
        """
        try:
            response = requests.get(f"{self.api_base}/devices/rubix", timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get('found'):
                logger.info(f"Rubix44 found: Input device {data['input_device']}, "
                          f"Output device {data['output_device']}")
                input_info = data.get('input_device_info', {})
                logger.info(f"  Input channels: {input_info.get('channels')}, "
                          f"Sample rate: {input_info.get('sample_rate')} Hz")
                return data
            else:
                logger.warning("Rubix44 device not found on server")
                return None
        except Exception as e:
            logger.error(f"Error checking Rubix device: {e}")
            return None

    def get_playback_files(self) -> List[Dict]:
        """
        Get list of available playback files with metadata.

        Returns:
            List of playback file dictionaries with filename, duration, sample_rate, channels
        """
        response = requests.get(f"{self.api_base}/playback-files", timeout=10)
        response.raise_for_status()
        return response.json()

    def get_recording_history(self) -> List[Dict]:
        """
        Get list of all recorded sessions with enhanced v1.1.0 metadata.

        Returns:
            List of recording session dictionaries, each containing:
            - id: Session ID
            - prefix: Output prefix
            - timestamp: Recording timestamp
            - start_time: ISO format start time (v1.1.0)
            - end_time: ISO format end time (v1.1.0)
            - duration_seconds: Actual recording duration (v1.1.0)
            - playback_file: Stimulus file used (v1.1.0)
            - sample_rate: Recording sample rate (v1.1.0)
            - files: List of file dictionaries with name, path, size, modified
        """
        response = requests.get(f"{self.api_base}/recordings/history", timeout=10)
        response.raise_for_status()
        return response.json()

    def download_recording(self, filename: str, save_path: Path) -> bool:
        """
        Download a recording file from the server.

        Args:
            filename: Name of the file to download
            save_path: Local path where file should be saved

        Returns:
            True if download succeeded
        """
        try:
            # Use the recordings endpoint to download file
            response = requests.get(
                f"{self.api_base}/recordings/{filename}",
                timeout=300,  # 5 min timeout for large files
                stream=True
            )
            response.raise_for_status()

            # Create parent directory if needed
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Stream download to file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Downloaded {filename} to {save_path} ({save_path.stat().st_size} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False


class Rubix44DataProvider:
    """
    Data provider that polls rubix44-recorder for new recordings and processes them
    for the continuous learning pipeline.
    """

    def __init__(self,
                 api_url: str,
                 download_dir: Path,
                 processor,
                 database,
                 state_file: Optional[Path] = None,
                 output_prefix_filter: Optional[str] = None,
                 cleanup_after_processing: bool = False,
                 metadata_backend: str = 'mariadb',
                 validate_device_on_startup: bool = True,
                 min_recording_duration_sec: float = 60.0):
        """
        Initialize the rubix44 data provider.

        Args:
            api_url: URL of rubix44-recorder API
            download_dir: Directory where recordings should be downloaded
            processor: StereoChannelProcessor instance for feature extraction
            database: FeatureDatabase instance for storing features
            state_file: Path to JSON file tracking processed recordings (default: download_dir/.state.json)
            output_prefix_filter: Only process files with this prefix (None = all files)
            cleanup_after_processing: Delete WAV files after successful processing
            metadata_backend: Database backend for metadata ('mariadb' or 'sqlite')
            validate_device_on_startup: Verify Rubix44 device is connected (default: True)
            min_recording_duration_sec: Minimum recording duration to process (default: 60s)
        """
        self.client = Rubix44Client(api_url)
        self.download_dir = Path(download_dir)
        self.processor = processor
        self.database = database
        self.output_prefix_filter = output_prefix_filter
        self.cleanup_after_processing = cleanup_after_processing
        self.min_recording_duration_sec = min_recording_duration_sec

        # State file tracks which recordings we've already processed
        self.state_file = state_file or (self.download_dir / ".rubix44_state.json")
        self.processed_sessions: Set[str] = self._load_state()

        # MariaDB connection for metadata
        self.metadata_db = DatabaseConnection(backend=metadata_backend)

        # Create download directory
        self.download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Rubix44DataProvider initialized")
        logger.info(f"  API URL: {api_url}")
        logger.info(f"  Download dir: {self.download_dir}")
        logger.info(f"  Prefix filter: {output_prefix_filter or 'None (all files)'}")
        logger.info(f"  Min duration: {min_recording_duration_sec}s")
        logger.info(f"  Metadata backend: {metadata_backend}")
        logger.info(f"  Previously processed: {len(self.processed_sessions)} sessions")

        # Validate device on startup if requested
        if validate_device_on_startup:
            self._validate_rubix_device()

    def _validate_rubix_device(self):
        """Validate that Rubix44 device is connected and accessible."""
        try:
            device_info = self.client.get_rubix_device()
            if device_info:
                logger.info("Rubix44 device validated successfully")
            else:
                logger.warning(
                    "Rubix44 device not detected! Recordings may fail. "
                    "Check that the device is connected to the server."
                )
        except Exception as e:
            logger.error(f"Failed to validate Rubix44 device: {e}")

    def _load_state(self) -> Set[str]:
        """Load processed session IDs from state file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                return set(state.get('processed_sessions', []))
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return set()

    def _save_state(self):
        """Save processed session IDs to state file."""
        try:
            state = {
                'processed_sessions': list(self.processed_sessions),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state file: {e}")

    def check_health(self) -> bool:
        """
        Check if rubix44-recorder API is healthy.

        Returns:
            True if API is accessible
        """
        return self.client.health_check()

    def _get_metadata(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve metadata for a recording session from MariaDB.

        Args:
            session_id: Session ID to look up

        Returns:
            Dictionary with metadata if found and approved, None otherwise
        """
        try:
            with self.metadata_db.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)

                query = """
                    SELECT metadata_complete, quality_approved,
                           channel_1_expected_class, channel_2_expected_class,
                           channel_1_source, channel_2_source
                    FROM recording_sessions
                    WHERE session_id = %s
                """
                cursor.execute(query, (session_id,))
                metadata = cursor.fetchone()

                if not metadata:
                    logger.warning(f"No metadata found for session {session_id}")
                    return None

                # Check if metadata is complete and approved
                if not metadata.get('metadata_complete'):
                    logger.info(f"Session {session_id}: metadata not complete, skipping")
                    return None

                if not metadata.get('quality_approved'):
                    logger.info(f"Session {session_id}: quality not approved, skipping")
                    return None

                logger.info(f"Session {session_id}: metadata validated - "
                          f"Ch1={metadata['channel_1_source']}(class={metadata['channel_1_expected_class']}), "
                          f"Ch2={metadata['channel_2_source']}(class={metadata['channel_2_expected_class']})")

                return metadata

        except Exception as e:
            logger.error(f"Error fetching metadata for {session_id}: {e}", exc_info=True)
            return None

    def _update_processing_flags(self, session_id: str, imported: bool = True, processed: bool = True):
        """
        Update processing status flags in MariaDB after successful feature extraction.

        Args:
            session_id: Session ID to update
            imported: Set imported_to_features_db flag
            processed: Set processed_for_training flag
        """
        try:
            with self.metadata_db.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    UPDATE recording_sessions
                    SET imported_to_features_db = %s,
                        processed_for_training = %s
                    WHERE session_id = %s
                """
                cursor.execute(query, (imported, processed, session_id))
                conn.commit()

                logger.info(f"Updated processing flags for {session_id}: "
                          f"imported={imported}, processed={processed}")

        except Exception as e:
            logger.error(f"Error updating processing flags for {session_id}: {e}", exc_info=True)

    def poll_for_new_recordings(self) -> int:
        """
        Poll the API for new recordings and process them.

        Returns:
            Number of new recordings processed
        """
        logger.info("Polling for new recordings...")

        try:
            # Get recording history from API
            history = self.client.get_recording_history()
            logger.info(f"Found {len(history)} total sessions on server")

            # Filter for new sessions
            new_sessions = [
                session for session in history
                if session['id'] not in self.processed_sessions
            ]

            if not new_sessions:
                logger.info("No new recordings to process")
                return 0

            logger.info(f"Found {len(new_sessions)} new sessions to process")

            # Apply prefix filter if set
            if self.output_prefix_filter:
                filtered_sessions = [
                    s for s in new_sessions
                    if s.get('prefix', '').startswith(self.output_prefix_filter)
                ]
                logger.info(f"After prefix filter '{self.output_prefix_filter}': {len(filtered_sessions)} sessions")
                new_sessions = filtered_sessions

            # Process each new session
            processed_count = 0
            for session in new_sessions:
                if self._process_session(session):
                    processed_count += 1
                    self.processed_sessions.add(session['id'])
                    self._save_state()

            logger.info(f"Successfully processed {processed_count}/{len(new_sessions)} new sessions")
            return processed_count

        except Exception as e:
            logger.error(f"Error polling for recordings: {e}")
            return 0

    def _process_session(self, session: Dict) -> bool:
        """
        Process a single recording session with enhanced v1.1.0 metadata.

        Args:
            session: Session dictionary from API with v1.1.0 fields

        Returns:
            True if processing succeeded
        """
        session_id = session['id']

        # Extract v1.1.0 enhanced metadata
        duration_sec = session.get('duration_seconds', 0)
        playback_file = session.get('playback_file', 'unknown')
        sample_rate = session.get('sample_rate', 44100)
        start_time = session.get('start_time', 'unknown')
        end_time = session.get('end_time', 'unknown')

        logger.info(f"Processing session: {session_id}")
        logger.info(f"  Duration: {duration_sec:.1f}s")
        logger.info(f"  Playback file: {playback_file}")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Time range: {start_time} to {end_time}")

        # Validate recording duration
        if duration_sec > 0 and duration_sec < self.min_recording_duration_sec:
            logger.warning(
                f"Recording too short ({duration_sec:.1f}s < "
                f"{self.min_recording_duration_sec}s), skipping"
            )
            return False

        # Check metadata before processing
        metadata = self._get_metadata(session_id)
        if not metadata:
            logger.info(f"Skipping session {session_id}: no valid metadata")
            return False

        # Find stereo file (we only need the stereo file, not individual channels)
        stereo_file = None
        for file_info in session.get('files', []):
            if '_stereo.wav' in file_info['name']:
                stereo_file = file_info
                break

        if not stereo_file:
            logger.warning(f"No stereo file found for session {session_id}")
            return False

        filename = stereo_file['name']
        local_path = self.download_dir / filename

        try:
            # Download if not already present
            if not local_path.exists():
                logger.info(f"Downloading {filename}...")
                if not self.client.download_recording(filename, local_path):
                    logger.error(f"Download failed for {filename}")
                    return False
            else:
                logger.info(f"File already exists locally: {local_path}")

            # Validate file size
            expected_size = stereo_file.get('size', 0)
            actual_size = local_path.stat().st_size
            if expected_size > 0 and abs(actual_size - expected_size) > 1024:
                logger.warning(f"Size mismatch: expected {expected_size}, got {actual_size}")

            # Get expected class labels from metadata
            ch1_class = int(metadata['channel_1_expected_class'])
            ch2_class = int(metadata['channel_2_expected_class'])

            logger.info(f"Using labels from metadata: Ch1={ch1_class}, Ch2={ch2_class}")

            # Process stereo file to extract features with metadata-specified labels
            logger.info(f"Extracting features from {filename}...")
            X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
                self.processor.process_stereo_file(
                    local_path,
                    positive_label=ch1_class,
                    negative_label=ch2_class
                )

            # Store features in database with source tracking
            timestamp = self._parse_timestamp(session['timestamp'])

            logger.info(f"Storing {len(X_left)} channel 1 (class={ch1_class}) samples...")
            self.database.insert_features_batch(
                X=X_left,
                y=y_left,
                channel=1,  # Left channel
                source_file=filename,
                offsets=offsets_left,
                timestamp=timestamp
            )

            logger.info(f"Storing {len(X_right)} channel 2 (class={ch2_class}) samples...")
            self.database.insert_features_batch(
                X=X_right,
                y=y_right,
                channel=0,  # Right channel
                source_file=filename,
                offsets=offsets_right,
                timestamp=timestamp
            )

            logger.info(f"Successfully processed {session_id}: "
                       f"{len(X_left)} channel 1 (class={ch1_class}) + "
                       f"{len(X_right)} channel 2 (class={ch2_class}) samples")

            # Update processing flags in MariaDB
            self._update_processing_flags(session_id, imported=True, processed=True)

            # Cleanup if requested
            if self.cleanup_after_processing:
                logger.info(f"Deleting processed file: {local_path}")
                local_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}", exc_info=True)
            return False

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse timestamp from session ID format: YYYY-MM-DD_HH-MM-SS

        Args:
            timestamp_str: Timestamp string from session

        Returns:
            datetime object
        """
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
        except Exception as e:
            logger.warning(f"Could not parse timestamp '{timestamp_str}': {e}")
            return datetime.now()

    def get_stats(self) -> Dict:
        """
        Get statistics about processed recordings.

        Returns:
            Dictionary with provider statistics
        """
        return {
            'api_url': self.client.api_base,
            'download_dir': str(self.download_dir),
            'processed_sessions': len(self.processed_sessions),
            'prefix_filter': self.output_prefix_filter,
            'cleanup_enabled': self.cleanup_after_processing,
            'api_healthy': self.check_health()
        }
