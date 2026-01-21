#!/usr/bin/env python3
"""
Check metadata for a specific record ID.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

def check_id_in_all_tables(search_id):
    """Search for an ID across all tables with metadata"""

    db = DatabaseConnection(backend='mariadb')

    tables = [
        'continuous_experiments',
        'recording_sessions',
        'recording_cycles',
        'experiment_alerts',
        'substance_vocabulary',
        'pipeline_status'
    ]

    print(f"\nSearching for ID: {search_id}")
    print("=" * 80)

    found = False

    with db.get_connection() as conn:
        cursor = conn.cursor()

        for table in tables:
            # Get primary key column
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'xor_project'
                AND TABLE_NAME = %s
                AND COLUMN_KEY = 'PRI'
                LIMIT 1
            """, (table,))

            result = cursor.fetchone()
            if not result:
                continue

            pk_col = result[0]

            # Search for the ID
            cursor.execute(f"""
                SELECT {pk_col}, metadata
                FROM {table}
                WHERE {pk_col} LIKE %s
            """, (f"%{search_id}%",))

            results = cursor.fetchall()

            if results:
                found = True
                print(f"\n✓ Found in table: {table}")
                print("-" * 80)

                for record_id, metadata in results:
                    print(f"  Record ID: {record_id}")

                    if metadata:
                        print(f"  Has metadata: YES")
                        try:
                            meta = json.loads(metadata)
                            print(f"  User: {meta.get('user', 'N/A')}")
                            print(f"  Tags: {', '.join(meta.get('tags', []))}")
                            print(f"  Priority: {meta.get('priority', 'N/A')}")
                        except json.JSONDecodeError:
                            print(f"  Metadata (raw): {metadata[:100]}...")
                    else:
                        print(f"  Has metadata: NO ❌")
                        print(f"  Status: MISSING METADATA")

    if not found:
        print(f"\n❌ ID '{search_id}' not found in any table")
        print("\nSearching in all columns (not just primary keys)...")

        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Try recording_sessions with session_id
            for table in tables:
                cursor.execute(f"SHOW COLUMNS FROM {table}")
                columns = [col[0] for col in cursor.fetchall()]

                # Search in all string columns
                for col in columns:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {table}
                        WHERE {col} LIKE %s
                    """, (f"%{search_id}%",))

                    count = cursor.fetchone()[0]
                    if count > 0:
                        print(f"\n✓ Found {count} matches in {table}.{col}")

                        cursor.execute(f"""
                            SELECT * FROM {table}
                            WHERE {col} LIKE %s
                            LIMIT 5
                        """, (f"%{search_id}%",))

                        results = cursor.fetchall()
                        for row in results:
                            print(f"  Row: {row}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check metadata for specific ID')
    parser.add_argument('search_id', help='ID to search for')

    args = parser.parse_args()

    check_id_in_all_tables(args.search_id)
