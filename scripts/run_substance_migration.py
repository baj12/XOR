#!/usr/bin/env python3
"""
Run Substance Vocabulary Schema Migration

Executes the SQL migration script to refactor continuous_experiments
to use single substance field per channel instead of redundant source + class fields.

Usage:
    python scripts/run_substance_migration.py [--dry-run]

Author: Claude Code
Date: 2026-01-04
"""

import sys
import os
from pathlib import Path
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_connection import DatabaseConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_migration(dry_run: bool = False) -> bool:
    """
    Run the substance vocabulary migration.

    Args:
        dry_run: If True, only validate the SQL without executing

    Returns:
        True if successful, False otherwise
    """
    # Read migration SQL
    sql_file = Path(__file__).parent / 'migrate_substance_schema.sql'

    if not sql_file.exists():
        logger.error(f"Migration SQL file not found: {sql_file}")
        return False

    logger.info(f"Reading migration SQL from: {sql_file}")
    with open(sql_file, 'r') as f:
        sql_content = f.read()

    # Split into individual statements
    statements = []
    current_statement = []
    delimiter = ';'

    for line in sql_content.split('\n'):
        line = line.strip()

        # Handle DELIMITER changes
        if line.startswith('DELIMITER'):
            if current_statement:
                statements.append('\n'.join(current_statement))
                current_statement = []
            delimiter = line.split()[-1]
            continue

        # Skip comments and empty lines
        if not line or line.startswith('--'):
            continue

        current_statement.append(line)

        # Check for statement end
        if line.endswith(delimiter):
            stmt = '\n'.join(current_statement)
            if delimiter == '$$':
                stmt = stmt.rstrip('$$')
            else:
                stmt = stmt.rstrip(';')
            statements.append(stmt)
            current_statement = []

    # Add any remaining statement
    if current_statement:
        statements.append('\n'.join(current_statement))

    logger.info(f"Parsed {len(statements)} SQL statements")

    if dry_run:
        logger.info("DRY RUN - SQL statements that would be executed:")
        for i, stmt in enumerate(statements, 1):
            print(f"\n--- Statement {i} ---")
            print(stmt[:200] + "..." if len(stmt) > 200 else stmt)
        return True

    # Execute migration
    db = DatabaseConnection(backend='mariadb')

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)

            logger.info("Starting migration...")

            for i, stmt in enumerate(statements, 1):
                # Skip empty statements
                if not stmt.strip():
                    continue

                try:
                    logger.info(f"Executing statement {i}/{len(statements)}...")

                    # Show first part of statement for context
                    stmt_preview = stmt.strip().split('\n')[0][:100]
                    logger.debug(f"  {stmt_preview}...")

                    cursor.execute(stmt)

                    # Show results if any
                    if cursor.description:
                        results = cursor.fetchall()
                        if results:
                            logger.info(f"  Results: {len(results)} rows")
                            for row in results[:5]:  # Show first 5 rows
                                logger.info(f"    {row}")

                except Exception as e:
                    logger.error(f"Error executing statement {i}: {e}")
                    logger.error(f"Statement: {stmt[:500]}")
                    raise

            conn.commit()
            logger.info("Migration completed successfully!")

            # Verify migration
            logger.info("\nVerifying migration...")

            cursor.execute("SELECT COUNT(*) FROM substance_vocabulary")
            vocab_count = cursor.fetchone()[0]
            logger.info(f"  Substance vocabulary entries: {vocab_count}")

            cursor.execute("""
                SELECT COUNT(*) FROM continuous_experiments
                WHERE channel_1_substance IS NOT NULL OR channel_2_substance IS NOT NULL
            """)
            migrated_count = cursor.fetchone()[0]
            logger.info(f"  Experiments with substances: {migrated_count}")

            cursor.execute("""
                SELECT COUNT(*) FROM continuous_experiments
                WHERE faraday_cage_used = TRUE
            """)
            faraday_count = cursor.fetchone()[0]
            logger.info(f"  Experiments with Faraday cage: {faraday_count}")

            # Show sample migrated data
            logger.info("\nSample migrated experiments:")
            cursor.execute("""
                SELECT experiment_id, experiment_name, channel_1_substance,
                       channel_2_substance, faraday_cage_used
                FROM continuous_experiments
                LIMIT 3
            """)
            for row in cursor.fetchall():
                logger.info(f"  {row}")

            cursor.close()
            return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run substance vocabulary migration')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate SQL without executing')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Substance Vocabulary Schema Migration")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    success = run_migration(dry_run=args.dry_run)

    if success:
        logger.info("\n✓ Migration completed successfully!")
        if not args.dry_run:
            logger.info("\nNext steps:")
            logger.info("1. Update API endpoints to use substance vocabulary")
            logger.info("2. Update orchestrator to resolve class from substance")
            logger.info("3. Update web interface for substance selection")
        return 0
    else:
        logger.error("\n✗ Migration failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
