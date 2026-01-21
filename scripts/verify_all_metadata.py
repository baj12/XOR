#!/usr/bin/env python3
"""
Verify metadata has been added to all tables.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection

def check_table_metadata(cursor, table_name):
    """Check if table has metadata column and count populated records"""

    # Check if metadata column exists
    cursor.execute("""
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'xor_project'
        AND TABLE_NAME = %s
        AND COLUMN_NAME = 'metadata'
    """, (table_name,))

    has_column = cursor.fetchone()[0] > 0

    if not has_column:
        return {'has_column': False, 'total': 0, 'with_metadata': 0, 'without_metadata': 0}

    # Count total records
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total = cursor.fetchone()[0]

    # Count records with metadata
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE metadata IS NOT NULL AND metadata != ''
    """)
    with_metadata = cursor.fetchone()[0]

    without_metadata = total - with_metadata

    return {
        'has_column': True,
        'total': total,
        'with_metadata': with_metadata,
        'without_metadata': without_metadata
    }

def main():
    db = DatabaseConnection(backend='mariadb')

    # Tables to check
    tables_to_check = [
        'continuous_experiments',
        'recording_sessions',
        'recording_cycles',
        'experiment_alerts',
        'substance_vocabulary',
        'pipeline_status'
    ]

    print("\n" + "=" * 80)
    print("METADATA VERIFICATION REPORT")
    print("=" * 80)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        total_records = 0
        total_with_metadata = 0

        for table in tables_to_check:
            result = check_table_metadata(cursor, table)

            print(f"\nTable: {table}")
            print("-" * 80)

            if not result['has_column']:
                print("  ❌ No metadata column")
            else:
                total_records += result['total']
                total_with_metadata += result['with_metadata']

                status = "✅ COMPLETE" if result['without_metadata'] == 0 else "⚠️  INCOMPLETE"
                print(f"  {status}")
                print(f"  Total records:        {result['total']:6,}")
                print(f"  With metadata:        {result['with_metadata']:6,}")
                print(f"  Without metadata:     {result['without_metadata']:6,}")

                if result['total'] > 0:
                    percentage = (result['with_metadata'] / result['total']) * 100
                    print(f"  Completion:           {percentage:6.1f}%")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total records across all tables:  {total_records:6,}")
    print(f"Records with metadata:            {total_with_metadata:6,}")
    print(f"Records without metadata:         {total_records - total_with_metadata:6,}")

    if total_records > 0:
        overall_percentage = (total_with_metadata / total_records) * 100
        print(f"Overall completion:               {overall_percentage:6.1f}%")

        if overall_percentage == 100.0:
            print("\n🎉 ALL METADATA SUCCESSFULLY ADDED!")
        else:
            print(f"\n⚠️  {total_records - total_with_metadata} records still need metadata")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
