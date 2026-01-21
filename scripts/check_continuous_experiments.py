#!/usr/bin/env python3
"""
Check continuous_experiments table schema and data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

def main():
    db = DatabaseConnection(backend='mariadb')

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Get schema
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'xor_project'
            AND TABLE_NAME = 'continuous_experiments'
            ORDER BY ORDINAL_POSITION
        """)

        print("continuous_experiments Table Schema:")
        print("=" * 80)
        for row in cursor.fetchall():
            print(f"  {row[0]:30} {row[1]:15} NULL={row[2]:3} DEFAULT={row[3]}")

        print("\n")

        # Get data
        cursor.execute("""
            SELECT experiment_id, experiment_name, status, start_time, end_time
            FROM continuous_experiments
            ORDER BY experiment_id
        """)

        print("continuous_experiments Data:")
        print("=" * 80)
        for row in cursor.fetchall():
            print(f"ID: {row[0]}")
            print(f"  Name: {row[1]}")
            print(f"  Status: {row[2]}")
            print(f"  Start: {row[3]}")
            print(f"  End: {row[4]}")
            print()

        # Check for metadata column
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'xor_project'
            AND TABLE_NAME = 'continuous_experiments'
            AND COLUMN_NAME = 'metadata'
        """)

        has_metadata = cursor.fetchone()[0] > 0
        print(f"\nHas metadata column: {has_metadata}")

if __name__ == "__main__":
    main()
