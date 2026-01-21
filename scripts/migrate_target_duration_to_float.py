#!/usr/bin/env python3
"""
Database Migration: Change target_duration_weeks from INT to FLOAT

This migration fixes the bug where fractional weeks (e.g., 0.01) were
being truncated to integers (e.g., 0) when stored in the database.

Usage:
    python scripts/migrate_target_duration_to_float.py [--dry-run]

Options:
    --dry-run    Show what would be changed without applying changes
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_connection import DatabaseConnection
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_target_duration_to_float(dry_run: bool = False):
    """
    Migrate target_duration_weeks column from INT to FLOAT.

    Args:
        dry_run: If True, only show what would be changed without applying
    """
    db = DatabaseConnection(backend='mariadb')

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            # Check current column type
            cursor.execute("DESCRIBE continuous_experiments")
            columns = cursor.fetchall()

            target_col = None
            for col in columns:
                if col['Field'] == 'target_duration_weeks':
                    target_col = col
                    break

            if not target_col:
                logger.error("Column 'target_duration_weeks' not found!")
                return False

            logger.info(f"Current column definition: {target_col}")
            logger.info(f"  Type: {target_col['Type']}")
            logger.info(f"  Null: {target_col['Null']}")
            logger.info(f"  Default: {target_col['Default']}")

            # Check if already FLOAT
            if 'float' in target_col['Type'].lower() or 'decimal' in target_col['Type'].lower():
                logger.info("Column is already FLOAT/DECIMAL type. No migration needed.")
                return True

            # Show current values that would be affected
            cursor.execute("""
                SELECT experiment_id, experiment_name, target_duration_weeks
                FROM continuous_experiments
                ORDER BY created_at DESC
                LIMIT 10
            """)
            experiments = cursor.fetchall()

            logger.info(f"\nCurrent experiments (showing up to 10 most recent):")
            for exp in experiments:
                logger.info(f"  {exp['experiment_id']}: {exp['experiment_name']} - {exp['target_duration_weeks']} weeks")

            if dry_run:
                logger.info("\n[DRY RUN] Would execute:")
                logger.info("  ALTER TABLE continuous_experiments")
                logger.info("  MODIFY COLUMN target_duration_weeks FLOAT NOT NULL;")
                logger.info("\nNo changes applied. Run without --dry-run to apply migration.")
                return True

            # Apply migration
            logger.info("\nApplying migration...")
            cursor.execute("""
                ALTER TABLE continuous_experiments
                MODIFY COLUMN target_duration_weeks FLOAT NOT NULL
            """)
            conn.commit()

            # Verify the change
            cursor.execute("DESCRIBE continuous_experiments")
            columns = cursor.fetchall()

            for col in columns:
                if col['Field'] == 'target_duration_weeks':
                    logger.info(f"\n✓ Migration successful!")
                    logger.info(f"  New type: {col['Type']}")
                    break

            return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Migrate target_duration_weeks column from INT to FLOAT'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without applying changes'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Database Migration: target_duration_weeks INT → FLOAT")
    logger.info("=" * 70)

    if args.dry_run:
        logger.info("Running in DRY RUN mode - no changes will be applied")

    success = migrate_target_duration_to_float(dry_run=args.dry_run)

    if success:
        logger.info("\n✓ Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("\n✗ Migration failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
