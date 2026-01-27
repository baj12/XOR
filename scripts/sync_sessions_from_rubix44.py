#!/usr/bin/env python3
"""
Sync Recording Sessions from Rubix44 API to MariaDB

This script:
1. Fetches all sessions from rubix44-recorder API
2. Creates missing entries in MariaDB recording_sessions table
3. Links sessions to their parent experiments based on prefix pattern
4. Inherits class labels and metadata from the parent experiment
5. Auto-approves sessions for testing purposes (with --auto-approve flag)

Usage:
    python scripts/sync_sessions_from_rubix44.py --dry-run
    python scripts/sync_sessions_from_rubix44.py --auto-approve
    python scripts/sync_sessions_from_rubix44.py --auto-approve --fill-random
"""

import argparse
import logging
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import requests

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from db_connection import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_api_sessions(api_url: str = "http://10.0.0.58:5000") -> list:
    """Fetch all sessions from rubix44 API."""
    response = requests.get(f"{api_url}/api/v1/recordings/history", timeout=30)
    response.raise_for_status()
    return response.json()


def extract_experiment_id(session_id: str) -> str | None:
    """
    Extract experiment ID from session_id.

    Patterns:
    - continuous_expexp_f09e9c18_cycle50_2026-01-20_... -> exp_f09e9c18
    - test_12hr_expexp_8ca89843_cycle1_2026-01-15_... -> exp_8ca89843
    - test_4hr_expexp_47f4785d_cycle1_... -> exp_47f4785d
    """
    # Look for pattern: expexp_<8char_hex>
    match = re.search(r'expexp_([a-f0-9]{8})', session_id)
    if match:
        return f"exp_{match.group(1)}"
    return None


def parse_session_timestamp(session_id: str) -> datetime | None:
    """
    Parse timestamp from session_id.

    Pattern: ..._2026-01-20_15-30-45
    """
    match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$', session_id)
    if match:
        date_str = match.group(1)
        time_str = match.group(2).replace('-', ':')
        try:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def extract_cycle_number(session_id: str) -> int | None:
    """Extract cycle number from session_id."""
    match = re.search(r'cycle(\d+)_', session_id)
    if match:
        return int(match.group(1))
    return None


def map_beaker_role(role: str | None) -> str:
    """
    Map experiment beaker role to recording_sessions enum.

    recording_sessions allows: 'recording', 'instrument', 'empty', 'not_used'
    experiments may have: 'none', 'negative', 'recording', etc.
    """
    if role is None:
        return 'not_used'

    role = role.lower().strip()

    # Direct matches
    if role in ('recording', 'instrument', 'empty', 'not_used'):
        return role

    # Mappings
    mappings = {
        'none': 'not_used',
        'negative': 'empty',  # negative = control = empty beaker
        'positive': 'recording',
        'control': 'empty',
        'active': 'recording',
    }

    return mappings.get(role, 'not_used')


def get_experiment_metadata(db: DatabaseConnection, experiment_id: str) -> dict | None:
    """Get metadata from parent experiment."""
    with db.get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT experiment_id, output_prefix,
                   channel_1_substance, channel_2_substance,
                   beaker_1_role, beaker_1_content,
                   beaker_2_role, beaker_2_content,
                   faraday_cage_used, researcher_name
            FROM continuous_experiments
            WHERE experiment_id = %s
        """, (experiment_id,))
        result = cursor.fetchone()

        if result:
            # Map beaker roles to valid enum values
            result['beaker_1_role'] = map_beaker_role(result.get('beaker_1_role'))
            result['beaker_2_role'] = map_beaker_role(result.get('beaker_2_role'))

        return result


def determine_class_labels(experiment: dict) -> tuple[int, int]:
    """
    Determine class labels based on experiment metadata.

    Convention:
    - Channel 1 (left) with active substance (lavender, etc.) = class 1
    - Channel 2 (right) with control (empty, air) = class 0

    If unclear, defaults to ch1=1, ch2=0
    """
    ch1_substance = (experiment.get('channel_1_substance') or '').lower()
    ch2_substance = (experiment.get('channel_2_substance') or '').lower()

    # Control substances
    controls = {'empty', 'air', 'none', 'control', 'blank', ''}

    # If ch1 is control and ch2 is active, swap
    if ch1_substance in controls and ch2_substance not in controls:
        return 0, 1

    # Default: ch1=1 (active), ch2=0 (control)
    return 1, 0


def sync_sessions(api_url: str, dry_run: bool, auto_approve: bool, fill_random: bool):
    """Main sync function."""

    # Get sessions from API
    logger.info(f"Fetching sessions from {api_url}...")
    api_sessions = get_api_sessions(api_url)
    logger.info(f"Found {len(api_sessions)} sessions on rubix44 server")

    # Connect to MariaDB
    db = DatabaseConnection(backend='mariadb')

    # Get existing sessions
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM recording_sessions")
        existing_sessions = {row[0] for row in cursor.fetchall()}

    logger.info(f"Found {len(existing_sessions)} existing sessions in MariaDB")

    # Track statistics
    stats = {
        'created': 0,
        'updated': 0,
        'skipped': 0,
        'linked_to_experiment': 0,
        'auto_approved': 0,
        'class_labels_set': 0,
    }

    # Cache for experiment metadata
    experiment_cache = {}

    for session in api_sessions:
        session_id = session['id']
        is_new = session_id not in existing_sessions

        # Extract metadata from session
        experiment_id = extract_experiment_id(session_id)
        recording_date = parse_session_timestamp(session_id)
        cycle_number = extract_cycle_number(session_id)
        duration = session.get('duration_seconds', 0)
        sample_rate = session.get('sample_rate', 44100)

        # Get experiment metadata if linked
        experiment = None
        ch1_class, ch2_class = None, None

        if experiment_id:
            if experiment_id not in experiment_cache:
                experiment_cache[experiment_id] = get_experiment_metadata(db, experiment_id)
            experiment = experiment_cache[experiment_id]

            if experiment:
                ch1_class, ch2_class = determine_class_labels(experiment)

        # Fill random values if requested and no experiment found
        if fill_random and (ch1_class is None or ch2_class is None):
            ch1_class = random.choice([0, 1])
            ch2_class = 1 - ch1_class  # Opposite class
            logger.debug(f"  Random class labels for {session_id}: ch1={ch1_class}, ch2={ch2_class}")

        # Find stereo filename
        stereo_filename = None
        for f in session.get('files', []):
            if '_stereo.wav' in f['name']:
                stereo_filename = f['name']
                break

        if dry_run:
            action = "Would create" if is_new else "Would update"
            logger.info(f"{action} session {session_id}")
            logger.info(f"  Experiment: {experiment_id}, Classes: ch1={ch1_class}, ch2={ch2_class}")
            if is_new:
                stats['created'] += 1
            else:
                stats['updated'] += 1
            continue

        # Insert or update session
        with db.get_connection() as conn:
            cursor = conn.cursor()

            if is_new:
                # Create new session
                cursor.execute("""
                    INSERT INTO recording_sessions (
                        session_id, recording_date, duration_seconds, sample_rate,
                        experiment_id, cycle_number, stereo_filename,
                        channel_1_source, channel_2_source,
                        channel_1_expected_class, channel_2_expected_class,
                        beaker_1_role, beaker_1_content,
                        beaker_2_role, beaker_2_content,
                        faraday_cage_used, researcher_name,
                        metadata_complete, quality_approved,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                """, (
                    session_id,
                    recording_date,
                    duration,
                    sample_rate,
                    experiment_id,
                    cycle_number,
                    stereo_filename,
                    experiment.get('channel_1_substance') if experiment else None,
                    experiment.get('channel_2_substance') if experiment else None,
                    ch1_class,
                    ch2_class,
                    experiment.get('beaker_1_role') if experiment else None,
                    experiment.get('beaker_1_content') if experiment else None,
                    experiment.get('beaker_2_role') if experiment else None,
                    experiment.get('beaker_2_content') if experiment else None,
                    experiment.get('faraday_cage_used', 0) if experiment else 0,
                    experiment.get('researcher_name') if experiment else None,
                    1 if ch1_class is not None else 0,  # metadata_complete
                    1 if auto_approve else 0,  # quality_approved
                ))
                conn.commit()
                stats['created'] += 1
                logger.info(f"Created session {session_id}")

            else:
                # Update existing session with missing data
                updates = []
                params = []

                if experiment_id:
                    updates.append("experiment_id = %s")
                    params.append(experiment_id)
                    stats['linked_to_experiment'] += 1

                if ch1_class is not None:
                    updates.append("channel_1_expected_class = %s")
                    params.append(ch1_class)
                    updates.append("channel_2_expected_class = %s")
                    params.append(ch2_class)
                    stats['class_labels_set'] += 1

                if auto_approve:
                    updates.append("quality_approved = 1")
                    stats['auto_approved'] += 1

                if ch1_class is not None:
                    updates.append("metadata_complete = 1")

                if updates:
                    updates.append("updated_at = NOW()")
                    params.append(session_id)

                    query = f"UPDATE recording_sessions SET {', '.join(updates)} WHERE session_id = %s"
                    cursor.execute(query, params)
                    conn.commit()
                    stats['updated'] += 1
                    logger.info(f"Updated session {session_id}")
                else:
                    stats['skipped'] += 1

        if experiment_id:
            stats['linked_to_experiment'] += 1
        if ch1_class is not None:
            stats['class_labels_set'] += 1
        if auto_approve and is_new:
            stats['auto_approved'] += 1

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Sync Summary:")
    logger.info(f"  Sessions created: {stats['created']}")
    logger.info(f"  Sessions updated: {stats['updated']}")
    logger.info(f"  Sessions skipped: {stats['skipped']}")
    logger.info(f"  Linked to experiments: {stats['linked_to_experiment']}")
    logger.info(f"  Class labels set: {stats['class_labels_set']}")
    if auto_approve:
        logger.info(f"  Auto-approved: {stats['auto_approved']}")
    logger.info("=" * 60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Sync recording sessions from rubix44 API to MariaDB'
    )
    parser.add_argument(
        '--api-url',
        default='http://10.0.0.58:5000',
        help='Rubix44 API URL (default: http://10.0.0.58:5000)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--auto-approve',
        action='store_true',
        help='Auto-approve all sessions for processing'
    )
    parser.add_argument(
        '--fill-random',
        action='store_true',
        help='Fill random class labels for sessions without experiment linkage'
    )

    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    sync_sessions(
        api_url=args.api_url,
        dry_run=args.dry_run,
        auto_approve=args.auto_approve,
        fill_random=args.fill_random
    )


if __name__ == '__main__':
    main()
