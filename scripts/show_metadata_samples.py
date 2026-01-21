#!/usr/bin/env python3
"""
Show sample metadata from all tables with metadata.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

def get_primary_key(cursor, table_name):
    """Get the primary key column name for a table"""
    cursor.execute("""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'xor_project'
        AND TABLE_NAME = %s
        AND COLUMN_KEY = 'PRI'
        LIMIT 1
    """, (table_name,))

    result = cursor.fetchone()
    return result[0] if result else None

def show_sample(cursor, table_name, n=2):
    """Show sample metadata from a table"""

    pk_col = get_primary_key(cursor, table_name)
    if not pk_col:
        print(f"⚠️  Could not find primary key for {table_name}")
        return

    cursor.execute(f"""
        SELECT {pk_col}, metadata
        FROM {table_name}
        WHERE metadata IS NOT NULL
        LIMIT %s
    """, (n,))

    results = cursor.fetchall()

    if not results:
        print(f"⚠️  No metadata found in {table_name}")
        return

    print(f"\n{'=' * 80}")
    print(f"Table: {table_name} (showing {len(results)} of {n} samples)")
    print(f"{'=' * 80}")

    for pk_value, metadata_json in results:
        print(f"\nRecord ID: {pk_value}")
        print("-" * 80)

        try:
            metadata = json.loads(metadata_json)
            # Show compact version
            print(f"  User:        {metadata.get('user', 'N/A')}")
            print(f"  Tags:        {', '.join(metadata.get('tags', []))}")
            print(f"  Purpose:     {metadata.get('purpose', 'N/A')}")
            print(f"  Priority:    {metadata.get('priority', 'N/A')}")
            print(f"  Environment: {metadata.get('environment', 'N/A')}")
            if 'location' in metadata:
                print(f"  Location:    {metadata['location']}")
            if 'notes' in metadata:
                print(f"  Notes:       {metadata['notes']}")
        except json.JSONDecodeError:
            print(f"  ⚠️  Invalid JSON: {metadata_json[:100]}...")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Show sample metadata from all tables'
    )
    parser.add_argument('--samples', type=int, default=2,
                       help='Number of samples per table (default: 2)')
    parser.add_argument('--full', action='store_true',
                       help='Show full JSON instead of compact format')

    args = parser.parse_args()

    db = DatabaseConnection(backend='mariadb')

    tables = [
        'continuous_experiments',
        'recording_sessions',
        'recording_cycles',
        'experiment_alerts',
        'substance_vocabulary',
        'pipeline_status'
    ]

    print("\n" + "=" * 80)
    print("METADATA SAMPLES FROM ALL TABLES")
    print("=" * 80)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        for table in tables:
            show_sample(cursor, table, n=args.samples)

    print("\n" + "=" * 80)
    print("End of samples")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
