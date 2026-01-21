#!/usr/bin/env python3
"""
Initialize Continuous Learning Database Schema

Creates tables for continuous recording experiments in MariaDB.

Usage:
    python scripts/init_continuous_schema.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection


def main():
    """Initialize continuous learning schema"""
    print("=" * 60)
    print("Continuous Learning Schema Initialization")
    print("=" * 60)
    print()

    # Connect to database
    print("Connecting to MariaDB...")
    db = DatabaseConnection(backend='mariadb')

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)

            # Read SQL file
            sql_file = Path(__file__).parent / 'init_continuous_schema.sql'
            print(f"Reading schema from: {sql_file}")

            with open(sql_file, 'r') as f:
                sql_script = f.read()

            # Split into individual statements (handle multi-statement)
            statements = []
            current_statement = []

            for line in sql_script.split('\n'):
                # Skip comments
                if line.strip().startswith('--'):
                    continue

                current_statement.append(line)

                # Check if statement is complete
                if ';' in line:
                    stmt = '\n'.join(current_statement)
                    if stmt.strip() and not stmt.strip().startswith('--'):
                        statements.append(stmt)
                    current_statement = []

            print(f"Found {len(statements)} SQL statements")
            print()

            # Execute statements
            for i, statement in enumerate(statements, 1):
                # Skip USE statements (connection already has database)
                if statement.strip().upper().startswith('USE'):
                    continue

                # Skip SET statements for foreign key check
                if 'SET @' in statement:
                    continue
                if 'PREPARE' in statement or 'EXECUTE' in statement or 'DEALLOCATE' in statement:
                    continue

                try:
                    # Print statement type
                    stmt_type = statement.strip().split()[0].upper()
                    if stmt_type in ['CREATE', 'ALTER', 'DROP']:
                        obj_type = statement.strip().split()[1].upper()
                        if 'IF NOT EXISTS' in statement.upper() or 'IF EXISTS' in statement.upper():
                            obj_name = statement.split('IF')[1].split()[2] if 'NOT EXISTS' in statement.upper() else statement.split('IF')[1].split()[1]
                        else:
                            obj_name = statement.strip().split()[2].split('(')[0]
                        print(f"[{i}/{len(statements)}] {stmt_type} {obj_type} {obj_name}...")
                    else:
                        print(f"[{i}/{len(statements)}] Executing {stmt_type}...")

                    cursor.execute(statement)
                    conn.commit()

                except Exception as e:
                    # Some errors are okay (e.g., table already exists, column already exists)
                    error_str = str(e).lower()
                    if any(x in error_str for x in ['already exists', 'duplicate', 'check that column']):
                        print(f"  ⚠ Skipping (already exists)")
                    else:
                        print(f"  ❌ Error: {e}")
                        # Don't fail - continue with other statements
                        continue

            print()
            print("=" * 60)
            print("✅ Schema initialization complete!")
            print("=" * 60)
            print()

            # Verify tables exist
            print("Verifying tables...")
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'xor_project'
                AND TABLE_NAME IN ('continuous_experiments', 'recording_cycles', 'experiment_alerts')
                ORDER BY TABLE_NAME
            """)

            tables = cursor.fetchall()
            print(f"Found {len(tables)} continuous learning tables:")
            for table in tables:
                print(f"  ✓ {table[0]}")

            print()

            # Show views
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.VIEWS
                WHERE TABLE_SCHEMA = 'xor_project'
                AND TABLE_NAME IN ('experiment_summary', 'recent_cycles')
                ORDER BY TABLE_NAME
            """)

            views = cursor.fetchall()
            if views:
                print(f"Found {len(views)} views:")
                for view in views:
                    print(f"  ✓ {view[0]}")
                print()

            # Check recording_sessions extensions
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'xor_project'
                AND TABLE_NAME = 'recording_sessions'
                AND COLUMN_NAME IN ('experiment_id', 'cycle_number', 'auto_qc_passed')
                ORDER BY COLUMN_NAME
            """)

            columns = cursor.fetchall()
            if columns:
                print(f"Extended recording_sessions with {len(columns)} columns:")
                for col in columns:
                    print(f"  ✓ {col[0]}")
                print()

            print("Schema ready for continuous learning experiments!")
            print()

    except Exception as e:
        print(f"❌ Error initializing schema: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
