#!/usr/bin/env python3
"""
Check which experiments are missing metadata and their current schema.
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

        # First, check if metadata column exists
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'xor_project'
            AND TABLE_NAME = 'experiments'
            ORDER BY ORDINAL_POSITION
        """)

        print("Experiments Table Schema:")
        print("-" * 80)
        for row in cursor.fetchall():
            print(f"  {row[0]:25} {row[1]:15} NULL={row[2]:3} DEFAULT={row[3]}")

        print("\n")

        # Check total experiments
        cursor.execute("SELECT COUNT(*) FROM experiments")
        total = cursor.fetchone()[0]
        print(f"Total experiments: {total}")

        # Check if metadata column exists
        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = 'xor_project'
            AND TABLE_NAME = 'experiments'
            AND COLUMN_NAME = 'metadata'
        """)

        has_metadata_col = cursor.fetchone()[0] > 0

        if has_metadata_col:
            # Check for missing/empty metadata
            cursor.execute("""
                SELECT COUNT(*) FROM experiments
                WHERE metadata IS NULL OR metadata = ''
            """)
            missing = cursor.fetchone()[0]
            print(f"Experiments with missing/empty metadata: {missing}")

            # Show some examples
            cursor.execute("""
                SELECT experiment_id, description, experiment_type, status
                FROM experiments
                WHERE metadata IS NULL OR metadata = ''
                LIMIT 10
            """)

            if cursor.rowcount > 0:
                print("\nSample experiments missing metadata:")
                print("-" * 80)
                for row in cursor.fetchall():
                    print(f"  ID: {row[0]}, Type: {row[2]}, Status: {row[3]}")
                    print(f"      Desc: {row[1][:60] if row[1] else 'None'}...")
        else:
            print("\n⚠️  'metadata' column does NOT exist in experiments table!")
            print("   Need to add it first before populating.")

if __name__ == "__main__":
    main()
