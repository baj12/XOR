#!/usr/bin/env python3
"""
Transfer recordings from rubix44-recorder server to local CIH storage

This script:
1. Fetches recording list from rubix44-recorder API
2. Downloads recordings to local CIH storage (/Volumes/CIH/mora/moraWav/rubix/)
3. Updates MariaDB with recording metadata
4. Optionally deletes recordings from rubix server after successful transfer

Usage:
    # Transfer all recordings
    python scripts/transfer_rubix_recordings.py

    # Transfer and delete from server
    python scripts/transfer_rubix_recordings.py --delete-after-transfer

    # Transfer specific session
    python scripts/transfer_rubix_recordings.py --session-id recording_2026-01-04_18-11-06

    # Dry run (no actual transfer)
    python scripts/transfer_rubix_recordings.py --dry-run
"""

import sys
import argparse
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from db_connection import DatabaseConnection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RubixRecordingTransfer:
    """Handles transfer of recordings from rubix44-recorder to local storage"""

    def __init__(self,
                 api_url: str = "http://10.0.0.58:5000",
                 destination_dir: Path = Path("/Volumes/CIH/mora/moraWav/rubix"),
                 db_backend: str = "mariadb"):
        """
        Initialize transfer manager.

        Args:
            api_url: URL of rubix44-recorder API
            destination_dir: Local directory to store recordings
            db_backend: Database backend for metadata ('mariadb' or 'sqlite')
        """
        self.api_url = api_url.rstrip('/')
        self.api_base = f"{self.api_url}/api/v1"
        self.destination_dir = Path(destination_dir)
        self.db = DatabaseConnection(backend=db_backend)

        # Create destination directory if needed
        self.destination_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"RubixRecordingTransfer initialized")
        logger.info(f"  API: {self.api_url}")
        logger.info(f"  Destination: {self.destination_dir}")
        logger.info(f"  Database: {db_backend}")

    def get_recordings(self, session_id: Optional[str] = None) -> List[Dict]:
        """
        Get list of recordings from API.

        Args:
            session_id: Optional specific session to fetch

        Returns:
            List of recording session dictionaries
        """
        try:
            response = requests.get(f"{self.api_base}/recordings/history", timeout=10)
            response.raise_for_status()
            recordings = response.json()

            if session_id:
                recordings = [r for r in recordings if r['id'] == session_id]

            logger.info(f"Found {len(recordings)} recording(s)")
            return recordings

        except Exception as e:
            logger.error(f"Error fetching recordings: {e}")
            return []

    def download_file(self, filename: str, save_path: Path) -> bool:
        """
        Download a file from the API.

        Args:
            filename: Name of file to download
            save_path: Local path to save file

        Returns:
            True if download succeeded
        """
        try:
            logger.info(f"Downloading {filename}...")

            response = requests.get(
                f"{self.api_base}/recordings/{filename}",
                timeout=300,  # 5 min timeout
                stream=True
            )
            response.raise_for_status()

            # Stream to file
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = save_path.stat().st_size
            logger.info(f"  Downloaded: {save_path.name} ({file_size:,} bytes)")
            return True

        except Exception as e:
            logger.error(f"  Failed to download {filename}: {e}")
            return False

    def store_metadata_in_db(self, session: Dict, local_files: List[Path]) -> bool:
        """
        Store recording metadata in MariaDB recording_sessions table.

        Args:
            session: Session dictionary from API
            local_files: List of local file paths

        Returns:
            True if metadata stored successfully
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()

                # Extract metadata
                session_id = session['id']
                prefix = session.get('prefix', 'unknown')
                start_time_str = session.get('start_time', '')
                end_time_str = session.get('end_time', '')
                duration_sec = session.get('duration_seconds', 0)
                playback_file = session.get('playback_file', '')
                sample_rate = session.get('sample_rate', 44100)

                # Parse timestamps
                try:
                    # Handle both formats: "2026:01:04T18:11:06" and "2026-01-04T18:11:06"
                    start_time_str = start_time_str.replace(':', '-', 2)
                    recording_date = datetime.fromisoformat(start_time_str)
                except:
                    recording_date = datetime.now()

                # Find file paths
                stereo_file = None
                ch1_file = None
                ch2_file = None
                total_size = 0

                for f in local_files:
                    if '_stereo.wav' in f.name:
                        stereo_file = f.name
                    elif '_ch1.wav' in f.name:
                        ch1_file = f.name
                    elif '_ch2.wav' in f.name:
                        ch2_file = f.name
                    total_size += f.stat().st_size

                # Check if session already exists
                cursor.execute(
                    "SELECT session_id FROM recording_sessions WHERE session_id = %s",
                    (session_id,)
                )
                exists = cursor.fetchone()

                if exists:
                    logger.info(f"  Metadata already exists for {session_id}, updating...")
                    query = """
                        UPDATE recording_sessions
                        SET recording_date = %s,
                            duration_seconds = %s,
                            sample_rate = %s,
                            stereo_filename = %s,
                            ch1_filename = %s,
                            ch2_filename = %s,
                            file_size_bytes = %s
                        WHERE session_id = %s
                    """
                    cursor.execute(query, (
                        recording_date,
                        duration_sec,
                        sample_rate,
                        stereo_file,
                        ch1_file,
                        ch2_file,
                        total_size,
                        session_id
                    ))
                else:
                    logger.info(f"  Inserting new metadata for {session_id}...")
                    query = """
                        INSERT INTO recording_sessions
                        (session_id, recording_date, duration_seconds, sample_rate,
                         stereo_filename, ch1_filename, ch2_filename, file_size_bytes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(query, (
                        session_id,
                        recording_date,
                        duration_sec,
                        sample_rate,
                        stereo_file,
                        ch1_file,
                        ch2_file,
                        total_size
                    ))

                conn.commit()
                logger.info(f"  ✓ Metadata stored in database")
                return True

        except Exception as e:
            logger.error(f"  Error storing metadata: {e}")
            return False

    def delete_from_server(self, session_id: str) -> bool:
        """
        Delete recording from rubix server using API.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deletion succeeded
        """
        try:
            logger.info(f"Deleting {session_id} from server...")

            response = requests.post(
                f"{self.api_base}/recordings/delete",
                json={"session_id": session_id},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get('success'):
                deleted_count = result.get('deleted_count', 0)
                logger.info(f"  ✓ Deleted {deleted_count} file(s) from server")
                return True
            else:
                logger.error(f"  Delete failed: {result}")
                return False

        except Exception as e:
            logger.error(f"  Error deleting from server: {e}")
            return False

    def transfer_recording(self,
                          session: Dict,
                          delete_after: bool = False,
                          dry_run: bool = False) -> bool:
        """
        Transfer a single recording session.

        Args:
            session: Session dictionary from API
            delete_after: Delete from server after successful transfer
            dry_run: Don't actually transfer, just log what would happen

        Returns:
            True if transfer succeeded
        """
        session_id = session['id']
        files = session.get('files', [])

        if not files:
            logger.warning(f"No files found for {session_id}")
            return False

        logger.info(f"\nTransferring {session_id}:")
        logger.info(f"  Files: {len(files)}")
        logger.info(f"  Duration: {session.get('duration_seconds', 0):.1f}s")
        logger.info(f"  Playback: {session.get('playback_file', 'unknown')}")

        if dry_run:
            logger.info("  [DRY RUN] Would transfer files:")
            for file_info in files:
                logger.info(f"    - {file_info['name']}")
            return True

        # Download each file
        downloaded_files = []
        for file_info in files:
            filename = file_info['name']
            local_path = self.destination_dir / filename

            # Skip if already exists
            if local_path.exists():
                logger.info(f"  File already exists: {filename}")
                downloaded_files.append(local_path)
                continue

            if self.download_file(filename, local_path):
                downloaded_files.append(local_path)
            else:
                logger.error(f"  Failed to download {filename}")
                return False

        # Store metadata in database
        if downloaded_files:
            self.store_metadata_in_db(session, downloaded_files)

        # Delete from server if requested
        if delete_after and len(downloaded_files) == len(files):
            self.delete_from_server(session_id)

        logger.info(f"✓ Successfully transferred {session_id}")
        return True

    def transfer_all(self,
                    session_id: Optional[str] = None,
                    delete_after: bool = False,
                    dry_run: bool = False) -> Dict:
        """
        Transfer recordings from server to local storage.

        Args:
            session_id: Optional specific session to transfer
            delete_after: Delete from server after successful transfer
            dry_run: Don't actually transfer, just log what would happen

        Returns:
            Dictionary with transfer statistics
        """
        recordings = self.get_recordings(session_id)

        if not recordings:
            logger.warning("No recordings to transfer")
            return {'success': 0, 'failed': 0, 'skipped': 0}

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        for session in recordings:
            try:
                if self.transfer_recording(session, delete_after, dry_run):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            except Exception as e:
                logger.error(f"Error transferring {session['id']}: {e}")
                stats['failed'] += 1

        logger.info(f"\n" + "=" * 70)
        logger.info(f"Transfer Summary:")
        logger.info(f"  Successful: {stats['success']}")
        logger.info(f"  Failed: {stats['failed']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info("=" * 70)

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Transfer recordings from rubix44-recorder to local CIH storage"
    )
    parser.add_argument(
        '--api-url',
        default='http://10.0.0.58:5000',
        help='Rubix44-recorder API URL (default: http://10.0.0.58:5000)'
    )
    parser.add_argument(
        '--destination',
        type=Path,
        default=Path('/Volumes/CIH/mora/moraWav/rubix'),
        help='Local destination directory (default: /Volumes/CIH/mora/moraWav/rubix)'
    )
    parser.add_argument(
        '--session-id',
        help='Transfer specific session only'
    )
    parser.add_argument(
        '--delete-after-transfer',
        action='store_true',
        help='Delete recordings from server after successful transfer'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be transferred without actually doing it'
    )
    parser.add_argument(
        '--db',
        choices=['mariadb', 'sqlite'],
        default='mariadb',
        help='Database backend (default: mariadb)'
    )

    args = parser.parse_args()

    # Validate destination exists
    if not args.dry_run and not args.destination.parent.exists():
        logger.error(f"Destination parent directory does not exist: {args.destination.parent}")
        logger.error("Is the CIH volume mounted?")
        return 1

    # Create transfer manager
    manager = RubixRecordingTransfer(
        api_url=args.api_url,
        destination_dir=args.destination,
        db_backend=args.db
    )

    # Transfer recordings
    stats = manager.transfer_all(
        session_id=args.session_id,
        delete_after=args.delete_after_transfer,
        dry_run=args.dry_run
    )

    # Exit with error code if any transfers failed
    return 0 if stats['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
