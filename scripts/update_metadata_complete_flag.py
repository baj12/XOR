#!/usr/bin/env python3
"""
Update metadata_complete flag to TRUE for all recordings that have metadata JSON.
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

        # First, check current status
        print("Checking current metadata_complete status...")
        print("=" * 80)

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN metadata_complete = 1 THEN 1 END) as complete,
                COUNT(CASE WHEN metadata_complete = 0 THEN 1 END) as incomplete,
                COUNT(CASE WHEN metadata IS NOT NULL AND metadata != '' THEN 1 END) as has_metadata
            FROM recording_sessions
        """)

        stats = cursor.fetchone()
        print(f"Total recordings: {stats[0]}")
        print(f"metadata_complete = TRUE: {stats[1]}")
        print(f"metadata_complete = FALSE: {stats[2]}")
        print(f"Has metadata JSON: {stats[3]}")
        print()

        # Find recordings with metadata but metadata_complete = FALSE
        cursor.execute("""
            SELECT session_id
            FROM recording_sessions
            WHERE (metadata IS NOT NULL AND metadata != '')
            AND metadata_complete = 0
        """)

        to_update = cursor.fetchall()
        update_count = len(to_update)

        if update_count == 0:
            print("✓ All recordings with metadata already have metadata_complete = TRUE")
            return

        print(f"Found {update_count} recordings with metadata but metadata_complete = FALSE")
        print()

        # Update metadata_complete flag
        print("Updating metadata_complete flag...")

        cursor.execute("""
            UPDATE recording_sessions
            SET metadata_complete = 1
            WHERE (metadata IS NOT NULL AND metadata != '')
            AND metadata_complete = 0
        """)

        updated = cursor.rowcount
        conn.commit()

        print(f"✓ Updated {updated} recordings")
        print()

        # Verify the update
        print("Verification:")
        print("-" * 80)

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN metadata_complete = 1 THEN 1 END) as complete,
                COUNT(CASE WHEN metadata_complete = 0 THEN 1 END) as incomplete
            FROM recording_sessions
        """)

        new_stats = cursor.fetchone()
        print(f"Total recordings: {new_stats[0]}")
        print(f"metadata_complete = TRUE: {new_stats[1]}")
        print(f"metadata_complete = FALSE: {new_stats[2]}")
        print()

        # Show sample updated records
        cursor.execute("""
            SELECT session_id, metadata_complete,
                   LENGTH(metadata) as metadata_length
            FROM recording_sessions
            WHERE metadata IS NOT NULL AND metadata != ''
            LIMIT 5
        """)

        print("Sample updated records:")
        print("-" * 80)
        for row in cursor.fetchall():
            print(f"  {row[0]}: metadata_complete={row[1]}, metadata_length={row[2]} chars")

        print()
        print("✓ All recordings with metadata now have metadata_complete = TRUE!")

if __name__ == "__main__":
    main()
