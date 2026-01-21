#!/usr/bin/env python3
"""
Cleanup Stale Experiments

Automatically detects and cleans up experiments that are marked as "running"
but have no actual orchestrator process (e.g., after system restart/hibernation).

Can be run:
- Manually: python scripts/cleanup_stale_experiments.py
- On startup: Called automatically by Flask on boot
- Via cron: For periodic checks

Author: Claude Code
Date: 2026-01-05
"""

import sys
import os
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_connection import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_process_running(pid: int) -> bool:
    """
    Check if a process is actually running (not zombie/defunct).

    Args:
        pid: Process ID to check

    Returns:
        True if process exists and is not a zombie, False otherwise
    """
    if not pid:
        return False

    try:
        # Signal 0 doesn't kill process, just checks if it exists
        os.kill(pid, 0)

        # Process exists, but check if it's a zombie
        # Read /proc/<pid>/stat on Linux, or use ps on Mac
        import subprocess
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'stat='],
            capture_output=True,
            text=True,
            timeout=2
        )

        if result.returncode == 0:
            stat = result.stdout.strip()
            # If status starts with 'Z', it's a zombie (defunct)
            if stat and stat[0] == 'Z':
                logger.debug(f"PID {pid} is zombie/defunct")
                return False
            return True
        else:
            # ps failed, process doesn't exist
            return False

    except (OSError, ProcessLookupError):
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout checking PID {pid}")
        return False
    except Exception as e:
        logger.warning(f"Error checking PID {pid}: {e}")
        return False


def cleanup_stale_experiments(dry_run: bool = False) -> dict:
    """
    Find and cleanup experiments with stale "running" status.

    Args:
        dry_run: If True, only report what would be done

    Returns:
        Dict with statistics: {cleaned: int, found: int, details: list}
    """
    db = DatabaseConnection(backend='mariadb')

    cleaned = 0
    found = 0
    details = []

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Find all experiments marked as running
            cursor.execute("""
                SELECT experiment_id, experiment_name, orchestrator_pid,
                       status, current_cycle, updated_at
                FROM continuous_experiments
                WHERE status = 'running'
            """)

            running_experiments = cursor.fetchall()

            logger.info(f"Found {len(running_experiments)} experiments marked as 'running'")

            for exp in running_experiments:
                found += 1
                exp_id = exp['experiment_id']
                pid = exp['orchestrator_pid']

                # Check if process is actually running
                if pid is None:
                    # No PID stored - definitely stale
                    reason = "No PID stored"
                    is_stale = True
                elif not is_process_running(pid):
                    # PID exists but process is dead
                    reason = f"Process {pid} not running"
                    is_stale = True
                else:
                    # Process is actually running - not stale
                    logger.info(f"✓ {exp_id}: Process {pid} is running")
                    is_stale = False
                    continue

                # Found a stale experiment
                logger.warning(f"⚠ {exp_id}: STALE - {reason}")

                detail = {
                    'experiment_id': exp_id,
                    'experiment_name': exp['experiment_name'],
                    'pid': pid,
                    'reason': reason,
                    'last_updated': str(exp['updated_at']),
                    'current_cycle': exp['current_cycle']
                }
                details.append(detail)

                if not dry_run:
                    # Update database to mark as stopped
                    cursor.execute("""
                        UPDATE continuous_experiments
                        SET status = 'stopped',
                            orchestrator_pid = NULL,
                            updated_at = NOW()
                        WHERE experiment_id = %s
                    """, (exp_id,))

                    conn.commit()
                    cleaned += 1
                    logger.info(f"  → Cleaned up: set status='stopped', cleared PID")

                    # Log alert
                    import json
                    cursor.execute("""
                        INSERT INTO experiment_alerts
                        (experiment_id, alert_type, severity, message, details)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        exp_id,
                        'other',  # Use 'other' since 'stale_process' not in ENUM
                        'warning',
                        f'Stale process cleanup: Experiment marked as stopped due to missing orchestrator process. {reason}',
                        json.dumps(detail)
                    ))
                    conn.commit()

        result = {
            'found': found,
            'cleaned': cleaned,
            'details': details
        }

        return result

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
        raise


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Cleanup stale continuous experiments')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stale Experiment Cleanup")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    result = cleanup_stale_experiments(dry_run=args.dry_run)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Experiments checked: {result['found']}")
    logger.info(f"Stale experiments found: {len(result['details'])}")

    if args.dry_run:
        logger.info(f"Would clean up: {len(result['details'])} experiments")
    else:
        logger.info(f"Cleaned up: {result['cleaned']} experiments")

    if result['details']:
        logger.info("\nDetails:")
        for detail in result['details']:
            logger.info(f"  - {detail['experiment_id']}: {detail['reason']}")
            logger.info(f"    Name: {detail['experiment_name']}")
            logger.info(f"    Last updated: {detail['last_updated']}")

    if not args.dry_run and result['cleaned'] > 0:
        logger.info("\n✓ Cleanup complete!")
        logger.info("Affected experiments have been set to 'stopped' status.")
        logger.info("Alerts have been logged for each cleaned experiment.")

    return 0 if not result['details'] else 1


if __name__ == '__main__':
    sys.exit(main())
