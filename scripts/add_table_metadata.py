#!/usr/bin/env python3
"""
General-purpose script to add metadata column to any table and populate with random data.
"""

import sys
import json
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

# Random metadata generators
USERS = ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank', 'bernd', 'researcher1', 'researcher2']
TAGS = ['baseline', 'optimization', 'validation', 'production', 'long-run', 'test',
        'noise-analysis', 'stereo-test', 'rubix44', 'continuous-learning', 'experiment',
        'debug', 'comparison', 'final-model']
PURPOSES = ['research', 'production', 'testing', 'benchmarking', 'validation', 'long-term-study', 'debugging']
PRIORITIES = ['low', 'medium', 'high', 'critical']
ENVIRONMENTS = ['dev', 'staging', 'production', 'test', 'lab']
LOCATIONS = ['lab-bench-1', 'lab-bench-2', 'faraday-cage', 'quiet-room', 'main-lab']
CONDITIONS = ['normal', 'noisy', 'quiet', 'controlled', 'variable']

def generate_random_metadata():
    """Generate random metadata JSON"""
    metadata = {
        'user': random.choice(USERS),
        'tags': random.sample(TAGS, k=random.randint(1, 3)),
        'purpose': random.choice(PURPOSES),
        'priority': random.choice(PRIORITIES),
        'environment': random.choice(ENVIRONMENTS),
        'run_id': f"run_{random.randint(10000, 99999)}",
        'version': f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,20)}"
    }

    # Add some optional fields randomly
    if random.random() > 0.5:
        metadata['cost_estimate'] = round(random.uniform(1.0, 500.0), 2)

    if random.random() > 0.6:
        metadata['location'] = random.choice(LOCATIONS)

    if random.random() > 0.7:
        metadata['collaborators'] = random.sample(USERS, k=random.randint(1, 2))

    if random.random() > 0.6:
        metadata['notes'] = random.choice([
            'Successful run with good results',
            'Some issues detected',
            'Excellent baseline established',
            'Testing new configuration',
            'Long-term stability test',
            'Validation of previous results'
        ])

    return json.dumps(metadata, indent=2)

def get_table_primary_key(cursor, table_name):
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

def add_metadata_column(db, table_name):
    """Add metadata column to specified table if it doesn't exist"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'xor_project'
            AND TABLE_NAME = %s
            AND COLUMN_NAME = 'metadata'
        """, (table_name,))

        if cursor.fetchone()[0] > 0:
            print(f"✓ 'metadata' column already exists in {table_name}")
            return True

        # Add the column
        print(f"Adding 'metadata' column to {table_name} table...")
        cursor.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN metadata TEXT COMMENT 'JSON metadata'
        """)
        print("✓ Column added successfully")
        return True

def populate_metadata(db, table_name, force=False):
    """Populate metadata for records that don't have it"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Get primary key
        pk_col = get_table_primary_key(cursor, table_name)
        if not pk_col:
            print(f"⚠️  Could not determine primary key for {table_name}")
            return 0

        # Get records without metadata
        if force:
            cursor.execute(f"SELECT {pk_col} FROM {table_name}")
            print(f"Populating ALL records in {table_name} with new random metadata...")
        else:
            cursor.execute(f"""
                SELECT {pk_col} FROM {table_name}
                WHERE metadata IS NULL OR metadata = ''
            """)
            print(f"Populating {table_name} records with missing metadata...")

        records = cursor.fetchall()

        if not records:
            print(f"No records in {table_name} need metadata updates")
            return 0

        print(f"Found {len(records)} records to update")

        # Update each record
        updated = 0
        for (pk_value,) in records:
            metadata = generate_random_metadata()

            cursor.execute(
                f"UPDATE {table_name} SET metadata = %s WHERE {pk_col} = %s",
                (metadata, pk_value)
            )
            updated += 1

            if updated % 10 == 0 or updated == len(records):
                print(f"  ✓ Updated {updated}/{len(records)} records...")

        print(f"\n✓ Successfully updated {updated} records in {table_name}")
        return updated

def list_tables(db):
    """List all tables in the database"""
    with db.get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT TABLE_NAME, TABLE_ROWS
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'xor_project'
            ORDER BY TABLE_NAME
        """)

        print("\nAvailable tables in xor_project database:")
        print("=" * 60)
        for table_name, row_count in cursor.fetchall():
            row_count_str = f"{row_count:,}" if row_count is not None else "N/A"
            print(f"  {table_name:40} {row_count_str:>8} rows")
        print("=" * 60)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Add and populate metadata column for any table in xor_project database'
    )
    parser.add_argument('table', nargs='?',
                       help='Table name to add metadata to')
    parser.add_argument('--list', action='store_true',
                       help='List all available tables')
    parser.add_argument('--force', action='store_true',
                       help='Force update ALL records with new random metadata')

    args = parser.parse_args()

    db = DatabaseConnection(backend='mariadb')

    if args.list:
        list_tables(db)
        return

    if not args.table:
        print("Error: Please specify a table name or use --list to see available tables")
        parser.print_help()
        return

    # Add column if needed
    add_metadata_column(db, args.table)

    # Populate metadata
    populate_metadata(db, args.table, force=args.force)

if __name__ == "__main__":
    main()
