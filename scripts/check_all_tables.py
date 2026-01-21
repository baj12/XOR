#!/usr/bin/env python3
"""
Check all tables in the xor_project database for row counts.
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

        # Get all tables
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'xor_project'
            ORDER BY TABLE_NAME
        """)

        tables = [row[0] for row in cursor.fetchall()]

        print("XOR Project Database - Table Summary")
        print("=" * 80)

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table:30} {count:10,} rows")

        print("=" * 80)

if __name__ == "__main__":
    main()
