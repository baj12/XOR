#!/usr/bin/env python3
"""
Initialize MariaDB schema for recording management system

This script creates all necessary tables for managing recording sessions,
metadata, weather data, QC visualizations, and pipeline monitoring.

Usage:
    python scripts/init_recording_schema.py
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_schema():
    """Initialize recording management schema in MariaDB"""
    logger.info("Initializing recording management schema in MariaDB...")

    # Read SQL schema file
    schema_file = Path(__file__).parent / 'init_recording_schema.sql'
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return False

    with open(schema_file, 'r') as f:
        sql_content = f.read()

    # Split into individual statements (split on semicolon, but be careful with stored procedures)
    statements = []
    current_statement = []
    in_create_view = False

    for line in sql_content.split('\n'):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('--'):
            continue

        # Track CREATE VIEW statements (they might have semicolons in them)
        if 'CREATE OR REPLACE VIEW' in line.upper():
            in_create_view = True

        current_statement.append(line)

        # End of statement
        if line.endswith(';'):
            if in_create_view and 'SELECT' in ' '.join(current_statement).upper():
                # Continue collecting VIEW definition until we hit the final semicolon
                in_create_view = False

            statement = ' '.join(current_statement)
            if statement.strip():
                statements.append(statement)
            current_statement = []

    logger.info(f"Found {len(statements)} SQL statements to execute")

    # Connect to database and execute
    db = DatabaseConnection(backend='mariadb')

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            for i, statement in enumerate(statements, 1):
                try:
                    # Extract statement type for logging
                    stmt_type = statement.split()[0:3]
                    stmt_preview = ' '.join(stmt_type)

                    logger.info(f"[{i}/{len(statements)}] Executing: {stmt_preview}...")
                    cursor.execute(statement)
                    conn.commit()

                except Exception as e:
                    # Some statements might fail if objects already exist
                    if 'already exists' in str(e).lower() or 'duplicate' in str(e).lower():
                        logger.warning(f"Skipped (already exists): {stmt_preview}")
                    else:
                        logger.error(f"Failed to execute statement: {stmt_preview}")
                        logger.error(f"Error: {e}")
                        # Continue with other statements

            logger.info("Schema initialization completed successfully!")

            # Verify tables were created
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Tables in database: {', '.join(tables)}")

            # Check recording-specific tables
            expected_tables = ['recording_sessions', 'qc_visualizations', 'pipeline_status', 'experiments']
            found_tables = [t for t in expected_tables if t in tables]
            logger.info(f"Recording management tables created: {', '.join(found_tables)}")

            if len(found_tables) == len(expected_tables):
                logger.info("✓ All recording management tables created successfully")
                return True
            else:
                missing = set(expected_tables) - set(found_tables)
                logger.warning(f"⚠ Missing tables: {', '.join(missing)}")
                return False

    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def verify_schema():
    """Verify the schema is properly initialized"""
    logger.info("\nVerifying schema...")

    db = DatabaseConnection(backend='mariadb')

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Check recording_sessions table structure
            cursor.execute("DESCRIBE recording_sessions")
            columns = cursor.fetchall()
            logger.info(f"recording_sessions table has {len(columns)} columns")

            # Check for critical columns
            column_names = [col[0] for col in columns]
            critical_cols = ['session_id', 'channel_1_expected_class', 'channel_2_expected_class',
                           'weather_temperature_c', 'metadata_complete', 'quality_approved']

            for col in critical_cols:
                if col in column_names:
                    logger.info(f"  ✓ Column '{col}' exists")
                else:
                    logger.error(f"  ✗ Column '{col}' missing!")

            # Check pipeline_status
            cursor.execute("SELECT component, status FROM pipeline_status")
            status_rows = cursor.fetchall()
            logger.info(f"\nPipeline status initialized with {len(status_rows)} components:")
            for component, status in status_rows:
                logger.info(f"  - {component}: {status}")

            # Check experiments
            cursor.execute("SELECT experiment_id, experiment_name FROM experiments")
            exp_rows = cursor.fetchall()
            if exp_rows:
                logger.info(f"\nExperiments initialized: {len(exp_rows)}")
                for exp_id, exp_name in exp_rows:
                    logger.info(f"  - {exp_id}: {exp_name}")

            logger.info("\n✓ Schema verification completed")
            return True

    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 70)
    print("Recording Management System - Schema Initialization")
    print("=" * 70)
    print()

    success = init_schema()

    if success:
        print()
        verify_schema()
        print()
        print("=" * 70)
        print("✓ Schema initialization completed successfully!")
        print("=" * 70)
        sys.exit(0)
    else:
        print()
        print("=" * 70)
        print("✗ Schema initialization failed - check logs above")
        print("=" * 70)
        sys.exit(1)
